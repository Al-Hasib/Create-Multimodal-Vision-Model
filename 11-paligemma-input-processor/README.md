# Episode 11 — PaliGemma architecture review & the input processor

> **Video chapters:** `02:18:30 – 02:40:56` (PaliGemma Architecture review · PaliGemma input processor)
> **Target runtime:** 20–22 min · **Code:** [`code/processing_paligemma.py`](code/processing_paligemma.py), [`code/check_processor.py`](code/check_processor.py)

The vision tower is finished. Before writing any Gemma code, let's see the whole
architecture at once — then build the layer that turns a PIL image and a Python string into
tensors.

## The architecture, end to end

```
    image (any size)                          prompt: "this building is"
          │                                              │
    ┌─────▼──────────────────────────┐                    │
    │ processing_paligemma.py        │                    │
    │  resize 224x224 -> rescale     │      "<image>...<image><bos>this building is\n"
    │  -> normalize -> CHW           │                    │
    └─────┬──────────────────────────┘              tokenizer
          │ pixel_values [1,3,224,224]                     │ input_ids [1, 261]
          │                                                │
    ┌─────▼──────────────┐                    ┌────────────▼────────────┐
    │ SiglipVisionModel  │                    │ embed_tokens (Gemma)    │
    │ (Episodes 06-10)   │                    │ [Vocab, 2048]           │
    └─────┬──────────────┘                    └────────────┬────────────┘
          │ [1, 256, 1152]                                 │ [1, 261, 2048]
    ┌─────▼─────────────────────┐                          │
    │ MultiModalProjector       │                          │
    │ Linear(1152 -> 2048)      │                          │
    └─────┬─────────────────────┘                          │
          │ [1, 256, 2048]                                 │
          └──────────────► merge: overwrite the ◄──────────┘
                           <image> positions (Episode 13)
                                     │ [1, 261, 2048]
                           ┌─────────▼──────────┐
                           │ Gemma decoder x18  │  RMSNorm, GQA, RoPE, KV-cache
                           │ (Episodes 14-19)   │
                           └─────────┬──────────┘
                                     │ [1, 261, 2048]
                                lm_head (tied)
                                     │ [1, 261, 257152]
                              take logits[:, -1, :] -> sample -> next token
```

The essential design decision: **the image becomes tokens.** There is no cross-attention,
no adapter stack, no gated fusion. 256 image embeddings are spliced into the text sequence
and Gemma treats them like any other token. That is why PaliGemma is a good first VLM to
build — the multimodal part is one linear layer and one `masked_scatter`.

## The prompt format

```python
def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
```

For a 4-token image: `'<image><image><image><image><bos>this building is\n'`

Unusual, and every part is deliberate:

- **Image tokens first.** They are pure placeholders — the tokenizer maps them to id
  `256000` and Episode 13 overwrites their *embeddings*. The ids themselves never reach the
  language model as meaning.
- **`<bos>` after the image**, not at position 0. The text stream begins where the image
  ends.
- **A trailing `\n`.** PaliGemma was trained with it as part of the prompt format. It is not
  cosmetic: leave it out and the model's outputs get noticeably worse. The code comment
  notes that the paper suggests tokenizing it separately while the HF implementation does
  not — we follow HF, because we load HF weights.
- **No `<eos>`.** We are prefilling a prompt to continue, not scoring a finished sequence.
  Hence `tokenizer.add_bos_token = False` and `add_eos_token = False`: the processor
  controls the special tokens itself.

## The image pipeline

`process_images` is four steps, all written by hand rather than pulled from torchvision:

```python
resize(image, (224, 224), resample=BICUBIC)   # PIL, aspect ratio NOT preserved
np.array(image)                                # HWC uint8
rescale(image, 1/255.0)                        # -> [0, 1] float32
normalize(image, mean=[0.5]*3, std=[0.5]*3)    # -> [-1, 1]
image.transpose(2, 0, 1)                       # HWC -> CHW
```

Two easy ways to get this wrong:

1. **The mean/std are `0.5`, not the ImageNet values.** `IMAGENET_STANDARD_MEAN` in this
   file is `[0.5, 0.5, 0.5]` — the name is inherited from HF and is misleading. SigLIP was
   trained on a plain 0.5/0.5 normalization, mapping `[0,1]` to `[-1,1]`. Substituting the
   familiar `[0.485, 0.456, 0.406]` shifts every input off distribution and degrades
   captions in a way that is very hard to trace.
2. **Aspect ratio is discarded.** A 640×480 photo is squashed into a square. That is what
   the checkpoint expects; "fixing" it by letterboxing changes the input distribution.

## The extra tokens

```python
tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
EXTRA_TOKENS  = [f"<loc{i:04d}>" for i in range(1024)]   # bounding boxes
EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]    # segmentation
tokenizer.add_tokens(EXTRA_TOKENS)
```

This is how PaliGemma does detection and segmentation: **as text.** Ask it to detect a cat
and it emits `<loc0341><loc0122><loc0788><loc0654> cat` — normalized coordinates as
vocabulary tokens. Nothing in the architecture is task specific; the vocabulary is. That is
why `vocab_size` is 257,152 = 256,000 text + 1,024 loc + 128 seg (+ `<image>`).

## The output

```python
{"pixel_values": [1, 3, 224, 224], "input_ids": [1, 261], "attention_mask": [1, 261]}
```

261 = 256 image + `<bos>` + 3 text tokens + `\n`, for this prompt.

## The two assertions

```python
assert len(images) == 1 and len(text) == 1     # processing_paligemma.py
assert tokenizer.padding_side == "right"       # utils.py
```

Single image, single prompt, right padding. Together they guarantee **no padding in
practice**, which is what lets Episode 15's attention mask be a block of zeros. Batched
inference is a genuine extension of this codebase, not an oversight — Episode 15's
exercises trace exactly what would have to change.

## Run it

```bash
cd 11-paligemma-input-processor/code
python check_processor.py
```

This runs offline: the real PaliGemma tokenizer needs the gated download, so the check
supplies a minimal fake tokenizer with the same interface. The image pipeline is the real
code, and you can see the `[-1, 1]` range and the `[3, 224, 224]` layout in the output.

## Key takeaways

- Image → tokens; no cross-attention anywhere in PaliGemma.
- Prompt format: `<image> * N`, then `<bos>`, then the text, then `\n` — all four matter.
- Preprocessing: resize (aspect ratio dropped) → `/255` → normalize with 0.5/0.5 → CHW.
- `<locNNNN>` / `<segNNN>` make detection and segmentation plain text generation.
- One image, one prompt, right padding — by assertion.

## Gotchas

- **Using real ImageNet mean/std** because the constant is *named* `IMAGENET_STANDARD_MEAN`.
- **Forgetting the trailing `\n`** — silent quality loss, no error.
- **Putting `<bos>` before the image tokens** — plausible, wrong, and the model will still
  produce fluent text, so you may not notice.
- **`num_image_tokens` disagreeing with `(image_size / patch_size)²`** → Episode 13's
  `masked_scatter` mismatches and either raises or, worse, mis-assigns.
- **Left padding** breaks both the mask and the position ids.

## Exercises

1. Remove the `\n` and compare generated captions once you have real weights (Episode 20).
2. Set `num_image_tokens=128` while the tower produces 256 and follow the crash. Which
   assert or shape mismatch catches you first?
3. Swap `BICUBIC` for `NEAREST` and measure the mean absolute difference of the tensor.
4. Write a `process_images` variant that letterboxes to preserve aspect ratio, then compare
   captions on a very wide image. Does matching the training distribution beat preserving
   the geometry?
5. Decode `<loc0341>`'s id and work out the mapping from the 1024 loc tokens to normalized
   image coordinates.

## Further reading

- Beyer et al., *PaliGemma: A versatile 3B VLM for transfer*,
  <https://arxiv.org/abs/2407.07726>
- HF blog, *PaliGemma — Google's Cutting-Edge Open Vision Language Model*,
  <https://huggingface.co/blog/paligemma>

**Previous:** [Episode 10](../10-multi-head-attention-code/) ·
**Next:** [Episode 12 — Gemma config & weight tying](../12-gemma-config-weight-tying/)
