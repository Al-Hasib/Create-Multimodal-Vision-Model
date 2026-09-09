# Episode 20 — Inference, Top-P sampling, and running the model

> **Video chapters:** `05:23:40 – 05:43:40` (Inference code · Top-P Sampling `05:32:50` · Inference code `05:40:40` · Conclusion `05:43:40`)
> **Target runtime:** 20–25 min · **Code:** [`code/inference.py`](code/inference.py), [`code/utils.py`](code/utils.py), [`code/top_p_demo.py`](code/top_p_demo.py), [`code/smoke_test_tiny_model.py`](code/smoke_test_tiny_model.py)

The model is finished. This episode writes the code that loads real weights and generates
text — and then runs it.

## Loading the weights (`utils.py`)

```python
def load_hf_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = PaliGemmaConfig(**json.load(f))

    model = PaliGemmaForConditionalGeneration(config).to(device)
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()
    return (model, tokenizer)
```

The only place `transformers` is used at all — for the tokenizer, because SentencePiece
vocabularies are not something to reimplement in a tutorial.

- **`safetensors`** is a flat `name -> tensor` mapping, so loading is a dictionary build.
  The weights are split across several files; we merge them.
- **`strict=False`** because `lm_head.weight` is absent from the checkpoint (Episode 12) and
  our `position_ids` buffers are `persistent=False`. It is also why every module name had to
  match exactly: `strict=False` means a typo loads *nothing* for that module and reports no
  error.
- **`tie_weights()` after loading** creates the head by aliasing `embed_tokens`.

Worth doing once: print `set(tensors) - set(model.state_dict())` and the reverse. Both should
be nearly empty. Anything in there is a weight you are not using or a parameter that stayed
random.

## The generation loop (`inference.py`)

```python
kv_cache = KVCache()
stop_token = processor.tokenizer.eos_token_id
generated_tokens = []

for _ in range(max_tokens_to_generate):
    outputs = model(input_ids=input_ids, pixel_values=pixel_values,
                    attention_mask=attention_mask, kv_cache=kv_cache)
    kv_cache = outputs["kv_cache"]
    next_token_logits = outputs["logits"][:, -1, :]

    if do_sample:
        next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = _sample_top_p(next_token_logits, top_p)
    else:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    next_token = next_token.squeeze(0)
    generated_tokens.append(next_token)
    if next_token.item() == stop_token:
        break

    input_ids = next_token.unsqueeze(-1)                                    # only the new token
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=...)], dim=-1)
```

Everything from Episodes 14–15 in ten lines:

- **`logits[:, -1, :]`** — always the prediction for the *next* token. During prefill we
  compute 261 positions' logits and use exactly one; that is unavoidable in this
  implementation (and what "prefill" means).
- **`input_ids = next_token.unsqueeze(-1)`** — the whole point of the cache. From step 2 on,
  the model sees a sequence of length 1.
- **`attention_mask` grows by one** each step, which is what advances `position_ids` via
  `cumsum` (Episode 15).
- **`kv_cache` round-trips** through the output dict; the model stays stateless.
- **Greedy vs sampled** is one branch: `argmax` or top-p.
- **The whole thing runs under `torch.no_grad()`** (see `main`), and the model is in
  `.eval()`.

## Top-P (nucleus) sampling

```python
def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p          # note the subtraction
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
```

Keep the smallest set of tokens whose probability mass reaches `p`, renormalize, sample from
that. Unlike top-k it **adapts**: a confident step keeps one or two tokens, an uncertain step
keeps many.

Three lines deserve attention:

**`probs_sum - probs_sort > p`.** Subtracting shifts the cumulative sum one place right, so
the comparison uses the mass *before* adding the current token. The token that crosses `p`
is therefore **kept**, and the kept set is never empty. With a plain `probs_sum > p` you cut
the crossing token, and for a very confident distribution (top token = 0.95, p = 0.9) you
would cut *everything*.

**`div_` renormalizes** — `torch.multinomial` needs a valid distribution.

**`torch.gather(probs_idx, -1, next_token)`.** `multinomial` returns an index into the
*sorted* array, not a token id. Forget the gather and your model generates fluent text from
entirely wrong ids.

Temperature is applied before all of this: `softmax(logits / temperature)`. `T < 1` sharpens,
`T > 1` flattens, `T → 0` approaches `argmax`. Note that this implementation passes
*probabilities* into `_sample_top_p`, not logits — worth knowing if you refactor.

## Running it for real

```bash
huggingface-cli download google/paligemma-3b-pt-224 \
    --local-dir ~/projects/paligemma-weights/paligemma-3b-pt-224
```

Then edit `launch_inference.sh`:

```bash
MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
PROMPT="this building is "
IMAGE_FILE_PATH="test_images/pic1.jpeg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"
```

`test_images/` is gitignored — create it and drop in your own image. Then:

```bash
./launch_inference.sh
```

Expect a couple of minutes on CPU for the first tokens (the 2.9 B parameters have to be read
per token) and near-instant output on a GPU. `pt` is the *base* model: it completes prompts
rather than answering questions, so `"this building is "` works better than
`"What is this?"`. Try `"caption en"`, `"describe en"`, or `"detect building"` — the
prefixes it was trained on.

## Before you download 6 GB

```bash
cd 20-inference-and-top-p-sampling/code
python smoke_test_tiny_model.py
```

This builds a tiny randomly initialised PaliGemma and runs the real generation loop. The
output is gibberish — that is fine, and it is the point: it verifies the *machinery*
(prefill, cache growth, position ids, feeding back one token) without any weights.

```
PREFILL step 0: input_ids [1, 8]  logits [1, 8, 257]  cache 0 -> 8   next id 108
decode  step 1: input_ids [1, 1]  logits [1, 1, 257]  cache 8 -> 9   next id 108
decode  step 2: input_ids [1, 1]  logits [1, 1, 257]  cache 9 -> 10  next id 108
```

If your real run produces garbage but this passes, the bug is in the *weights* — module
names, `tie_weights`, the RMSNorm `1.0 +`, RoPE's pairing, or the image scaling. That is a
much shorter list to check.

```bash
python top_p_demo.py     # temperature and top-p on a 10-token vocabulary
```

## Where to go next

You have a working VLM. Things worth building on top of it, roughly by increasing effort:

1. **Cache the image features** so the vision tower runs once instead of once per token
   (Episode 14, exercise 5). The easiest real speedup in the codebase.
2. **Batched inference**: a real attention mask, left or right padding handled properly,
   per-sequence stopping. Episode 15's exercise 3 maps out the work.
3. **A preallocated KV-cache** instead of `torch.cat` each step.
4. **Swap in `F.scaled_dot_product_attention`** in both attentions and measure memory and
   speed.
5. **Compare against Hugging Face's implementation** token by token on the same prompt — the
   real test of a from-scratch build. Any divergence points at one of the gotchas in this
   playlist.
6. **Fine-tune it.** The forward pass supports gradients; add a loss over the suffix tokens
   and a dataset of image/caption pairs. Note `inference.py`'s `# TODO: remove the labels`
   comment — the labels path was stripped out of this codebase.
7. **`paligemma-3b-pt-448`**: 1024 image tokens. Which configs change? What gets slower, and
   by how much?

## Key takeaways

- `safetensors` + `strict=False` + `tie_weights()`; module names are the contract.
- Prefill once with the full prompt, then feed back one token at a time with the cache.
- `logits[:, -1, :]` is the next-token prediction; grow `attention_mask` by one per step.
- Top-p keeps the smallest set reaching mass `p`; the `- probs_sort` shift keeps the
  crossing token, so the set is never empty.
- `multinomial` returns a *sorted* index — `gather` it back to a token id.
- Smoke-test the machinery with random weights before blaming the weights.

## Gotchas

- **Reusing a `KVCache` across prompts** → the previous conversation leaks in.
- **Forgetting to grow `attention_mask`** → `position_ids` stops advancing and the model
  keeps generating at the same position.
- **Feeding the full sequence every step** *and* using the cache → duplicated keys, quietly
  wrong output.
- **Skipping `gather`** in top-p → fluent nonsense.
- **`probs_sum > p` without the shift** → an empty candidate set on confident steps.
- **Forgetting `torch.no_grad()` / `.eval()`** → memory blowup for no reason.
- **Asking a `pt` (base) checkpoint a question** and concluding your model is broken. Use a
  completion-style prompt, or an `-it` checkpoint.

## Conclusion

Roughly 900 lines of PyTorch, and every one of them is now a line you can explain:
patch embeddings, pre-norm residual layers, multi-head attention, a prefix-LM mask,
RMSNorm, GeGLU, grouped-query attention, rotary embeddings, a KV-cache, and nucleus
sampling. That set is not specific to PaliGemma — it is most of what a modern
transformer is made of, and reading a new model's code from here is mostly a matter of
spotting which of these pieces they swapped.

**Previous:** [Episode 19](../19-rotary-positional-embedding/) ·
**Back to:** [the playlist index](../README.md)
