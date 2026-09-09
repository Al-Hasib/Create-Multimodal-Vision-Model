# Episode 16 — GemmaModel, RMS normalization and the LM head

> **Video chapters:** `03:53:17 – 04:09:50` (Coding Gemma · RMS Normalization `04:02:45`)
> **Target runtime:** 16–18 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/rms_norm_demo.py`](code/rms_norm_demo.py), [`code/check_gemma_model.py`](code/check_gemma_model.py)

The language model's body: the embedding scale-up, the layer stack, the final norm, and the
head. Plus RMSNorm, which has a genuine trap in it.

## `GemmaModel`

```python
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, attention_mask, position_ids, inputs_embeds, kv_cache):
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, kv_cache=kv_cache,
            )
        return self.norm(hidden_states)
```

**Note the signature: it takes `inputs_embeds`, not `input_ids`.** Embedding happened back
in `PaliGemmaForConditionalGeneration.forward`, because the image features had to be spliced
in first (Episode 13). The `embed_tokens` table lives here but is *called* from outside via
`get_input_embeddings()`.

**`layer_idx` is passed to each layer** so its attention can address the right KV-cache slot
(Episode 14).

### The normalizer

```python
normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
hidden_states = hidden_states * normalizer
```

`sqrt(2048) ≈ 45.25`, applied to the entire merged sequence. This is Gemma's convention for
scaling token embeddings (the original transformer did the same thing), and it is the other
half of Episode 13's mystery division. Two related notes:

- It is built as a **tensor in the hidden states' dtype** on purpose. In bfloat16 the
  rounding of the constant itself is observable in the output, so matching the reference
  implementation exactly matters.
- It applies to *all* embeddings, image and text alike, which is precisely why the image
  features were pre-divided.

## `GemmaRMSNorm`

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))     # <- ZEROS

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())    # <- 1.0 +
        return output.type_as(x)
```

RMSNorm is LayerNorm with the mean subtraction and the bias removed:

```
LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm:    x            / sqrt(mean(x^2) + eps) * gamma
```

The claim of the RMSNorm paper — which held up — is that **re-scaling is the part that
matters**; re-centering contributes little. Dropping it saves a reduction pass and the bias
tensor. Small in FLOPs, real in memory traffic, and it runs 37 times per token in this model
(18 layers × 2 + 1 final). It also preserves the *direction* of the activation vector
exactly, where LayerNorm shifts it toward the origin.

### The `(1.0 + self.weight)` trap

`self.weight` is initialised to **zeros**, and the forward computes `output * (1.0 +
weight)`. So the effective scale starts at 1.0, exactly like LayerNorm's ones-initialised
weight — Gemma just stores `gamma - 1` instead of `gamma`.

Write `output * self.weight` (the "obvious" version, and what Llama does) with a Gemma
checkpoint and every normalized activation is multiplied by roughly zero. The model loads
cleanly and outputs garbage. This is a famous re-implementation bug; it is worth typing the
`1.0 +` deliberately and saying why.

### The `float()` dance

```python
output = self._norm(x.float())
output = output * (1.0 + self.weight.float())
return output.type_as(x)
```

`x.pow(2)` overflows float16 above about `x = 256`, so the statistics are always computed in
float32 and cast back at the end. The code comment records a second subtlety:

> Llama does `x.to(float16) * w` whilst Gemma is `(x * w).to(float16)`

Gemma multiplies in float32 and casts at the very end. A tiny numerical difference, faithful
to the checkpoint, and exactly the kind of detail that separates "my re-implementation is
close" from "my re-implementation matches".

## `GemmaForCausalLM`

```python
outputs = self.model(attention_mask=..., position_ids=..., inputs_embeds=..., kv_cache=...)
logits = self.lm_head(outputs).float()

return_data = {"logits": logits}
if kv_cache is not None:
    return_data["kv_cache"] = kv_cache
return return_data
```

- **Body / head split.** `GemmaModel` produces hidden states; `GemmaForCausalLM` adds the
  vocabulary head. Swap the head and the same body does classification.
- **`logits.float()`** — always compute the sampling softmax in float32 (Episode 03).
- **The cache is returned in the dict**, which is how `inference.py` threads it from step to
  step while the model itself stays stateless.
- The head is tied to `embed_tokens` (Episode 12), so a logit is a dot product between the
  hidden state and a token's embedding — the check script verifies this identity.

## Run it

```bash
cd 16-gemma-model-and-rms-norm/code
python rms_norm_demo.py
python check_gemma_model.py
```

`check_gemma_model.py` also computes the real parameter budget:

```
embedding table (shared with the head) :   526,647,296     (21% of the text model)
MLP  per layer (gate + up + down)      :   100,663,296
attention per layer (q,k,v,o)          :     9,437,184     (MLP is 11x bigger!)
x 18 layers + embedding                : 2,508,455,936     (~2.51B)
+ the vision tower                     :   412,442,352
= paligemma-3b                          : 2,920,898,288     (~2.92B)
```

## Key takeaways

- `GemmaModel.forward` takes `inputs_embeds`, because the image was merged upstream.
- All embeddings are multiplied by `sqrt(hidden_size)` on the way in — the counterpart to
  Episode 13's division.
- RMSNorm = LayerNorm without mean subtraction or bias; re-scaling is the part that matters.
- **`weight` is zeros and the forward uses `(1.0 + weight)`.**
- Statistics in float32, cast back at the end; Gemma casts *after* the multiply.
- `logits.float()` before sampling; the cache travels in the output dict.
- The MLP holds ~11× more parameters per layer than attention.

## Gotchas

- **`output * self.weight`** instead of `(1.0 + self.weight)` → near-zero activations,
  fluent garbage.
- **Initialising `weight` to ones** while also using `1.0 +` → everything doubled.
- **`mean(-1)` vs `sum(-1)`** in `_norm` → off by `sqrt(dim)`.
- **Skipping the `sqrt(hidden_size)` normalizer** → the whole model is off scale, and
  Episode 13's division now has nothing to cancel.
- **Computing the norm in float16** → overflow for activations above ~256.
- **Forgetting the final `self.norm`** → the last layer's residual output is never
  normalized before the head.

## Exercises

1. Change `(1.0 + self.weight)` to `self.weight` and print the hidden-state norm after one
   layer. Then do it with real weights in Episode 20 and read the output.
2. Implement LayerNorm and RMSNorm on the same input and compare cosine similarity between
   the input and each output. Which preserves direction?
3. Time both over 10,000 calls at `hidden_size=2048`. Is the saving worth a paper?
4. Remove the `.float()` calls and find an input magnitude where float16 RMSNorm produces
   `inf`.
5. Verify `lm_head(h) == h @ embed_tokens.weight.T` after tying, and explain what that means
   for how the model "chooses" a token.

## Further reading

- Zhang & Sennrich, *Root Mean Square Layer Normalization*,
  <https://arxiv.org/abs/1910.07467>
- Gemma Team, *Gemma: Open Models Based on Gemini Research and Technology*,
  <https://arxiv.org/abs/2403.08295>
- The `(x * w)` vs `x * w` cast difference: <https://github.com/huggingface/transformers/pull/29402>

**Previous:** [Episode 15](../15-attention-mask-and-position-ids/) ·
**Next:** [Episode 17 — Decoder layer & GeGLU FFN](../17-decoder-layer-and-geglu-ffn/)
