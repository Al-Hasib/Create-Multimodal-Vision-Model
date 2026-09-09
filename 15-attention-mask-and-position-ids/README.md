# Episode 15 — The attention mask, the position ids, and the image projection

> **Video chapters:** `03:33:35 – 03:53:17` (Coding Gemma · Image features projection `03:52:05`)
> **Target runtime:** 18–20 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/check_mask.py`](code/check_mask.py)

Write the real `KVCache`, finish `_merge_input_ids_with_image_features`, and complete
`PaliGemmaForConditionalGeneration.forward`. At the end of this episode the whole pipeline
runs end to end — with a pass-through language model.

## `KVCache`

Exactly as designed in Episode 14: two lists indexed by layer, growing along `dim=-2`,
with `num_items()` reading `key_cache[0].shape[-2]`. Nothing new — but now it is real, and
`num_items()` becomes the model's phase detector.

## The surprise: PaliGemma's prompt is not causally masked

```python
if kv_cache is None or kv_cache.num_items() == 0:
    # Prefill. Do not mask any token.
    causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
else:
    assert q_len == 1
    kv_len = kv_cache.num_items() + q_len
    causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

causal_mask = causal_mask.unsqueeze(1)   # [B, Q_Len, KV_Len] -> [B, 1, Q_Len, KV_Len]
```

**All zeros. In both branches. Nothing is ever masked.** If you expected a triangular
matrix, this is worth pausing on.

### Why the prefill is not causal

PaliGemma is a **prefix-LM**. The prompt — image tokens *and* text prompt — is a prefix, and
attention inside the prefix is **bidirectional**:

- Image tokens must see each other. An image is not a left-to-right sequence; this is the
  same argument that made the vision tower mask-free (Episode 05). Forcing causality on
  patch tokens would mean the top-left patch cannot see the rest of the picture.
- The prompt is *given*, never predicted. Nothing leaks by letting prompt token 5 attend to
  prompt token 7, because we never ask the model to predict token 7 from a prefix.

This is a real architectural choice from the PaliGemma paper, not an implementation
shortcut, and it is unusual enough that it is worth remembering when you read other VLMs.

### Why the decode step needs no mask either

During decoding `q_len == 1`: one new query, attending to the whole cache. Everything in the
cache is in the past by construction, so there is no future to hide. The `assert q_len == 1`
enforces the assumption that makes the all-zero mask valid.

### The caveat in the comment

```python
# This only works when we have no padding
```

With an all-zero mask, a padding position would be attended to exactly like a real token.
That is what the assertion in `forward` protects:

```python
assert torch.all(attention_mask == 1), "The input cannot be padded"
```

Together with Episode 11's single-image / single-prompt assertion, this is a consistent
(if restrictive) design: batch size 1, no padding, no mask. Batched inference means writing
a real mask — see the exercises.

### The head dimension

`unsqueeze(1)` inserts a broadcast axis: `[B, 1, Q_Len, KV_Len]` adds cleanly onto
`[B, Num_Heads_Q, Q_Len, KV_Len]` attention scores. One mask, all heads.

## Position ids

```python
if kv_cache is not None and kv_cache.num_items() > 0:
    # Decode: the query's position is just the last one.
    position_ids = attention_mask.cumsum(-1)[:, -1]
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
else:
    # Prefill: one position per token; masked slots get position 1.
    position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
```

**Why `cumsum` and not `arange`?** Because `cumsum` is padding-aware: a masked slot
contributes 0 and therefore does not advance the counter, so real tokens keep contiguous
positions regardless of padding. With no padding it reduces to `1, 2, 3, ...` — note
**1-based**, a direct consequence of using `cumsum`.

These indices are the *only* way the text model learns about order — they feed RoPE in
Episode 19. Get them wrong and the model still runs, still emits fluent text, and orders
things wrongly.

Note also that the attention mask coming *in* (from the processor, `[B, Seq]`, ones and
zeros) and the mask going *out* (`[B, 1, Q, KV]`, additive zeros) are different objects with
the same name. The input mask survives only as the source of the position ids.

## The complete forward pass

```python
def forward(self, input_ids, pixel_values, attention_mask=None, kv_cache=None):
    assert torch.all(attention_mask == 1), "The input cannot be padded"

    # 1. embed the text
    inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

    # 2. run the vision tower and project
    selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
    image_features = self.multi_modal_projector(selected_image_feature)

    # 3. merge, and build the mask + positions
    inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, attention_mask, kv_cache
    )

    # 4. language model
    return self.language_model(
        attention_mask=attention_mask, position_ids=position_ids,
        inputs_embeds=inputs_embeds, kv_cache=kv_cache,
    )
```

Eight lines for a multimodal model. Two details:

- **`pixel_values.to(inputs_embeds.dtype)`** — the processor produces float32; the model may
  be bfloat16. Without the cast, `Conv2d` raises a dtype mismatch.
- **The vision tower runs on every decode step**, recomputing identical image features.
  Harmless but wasteful — one of the best exercises in this repo (Episode 14, exercise 5).

## The image features projection

`03:52:05` in the video is a two-line topic that is worth naming explicitly: the projector
from Episode 12 sits between the tower and the merge, and the merge's
`/ sqrt(hidden_size)` from Episode 13 cancels `GemmaModel`'s input scale-up. Together:

```
vision tower [B,256,1152] --Linear--> [B,256,2048] --/sqrt(2048)--> merge --*sqrt(2048)--> Gemma
```

## Run it

```bash
cd 15-attention-mask-and-position-ids/code
python check_mask.py
```

The script prints the prefill mask (all zeros) next to what a *causal* mask would look
like, then manually seeds the cache to show the decode branch's `[1, 1, 1, 9]` shape and
single position id. Section 5 runs the full forward and produces real `logits`.

## Key takeaways

- PaliGemma is a prefix-LM: **no causal mask over image + prompt**. Deliberate.
- Decoding has `q_len == 1`, so the past is all there is — nothing to mask.
- The all-zero mask is only valid because padding is forbidden by assertion.
- `unsqueeze(1)` gives the mask a broadcast head axis.
- `position_ids` come from `attention_mask.cumsum(-1)` — padding-aware, 1-based.
- `kv_cache.num_items() == 0` is how the model knows it is in prefill.
- The full pipeline now runs: pixels → patches → projection → merge → mask/positions →
  language model → logits.

## Gotchas

- **Adding a causal mask "to be safe"** breaks the image tokens' bidirectional attention.
- **Using `arange` for the position ids** works here (no padding) and breaks the moment you
  batch.
- **0-based position ids** are off by one against the checkpoint's training. Subtle quality
  loss, no error.
- **Forgetting `unsqueeze(1)`** → the mask broadcasts against the head axis wrongly.
- **`-inf` instead of `finfo.min`** as a mask value → `nan` on a fully masked row.
- **Forgetting `pixel_values.to(inputs_embeds.dtype)`** → dtype error when running in
  bfloat16.

## Exercises

1. Remove `assert q_len == 1` and pass two new tokens during decode. What exactly is wrong
   with the mask that gets built?
2. Replace the prefill mask with a proper causal mask and compare captions after
   Episode 20. Which matches the HF implementation?
3. Batch two prompts of different lengths with right padding. List every line in
   `_merge_input_ids_with_image_features` and in `forward` that must change, and write the
   correct mask for that case.
4. Print `position_ids` for an `attention_mask` of `[1,1,0,0,1]` and explain each value.
5. Instrument `forward` to count vision-tower calls during a 20-token generation. Then cache
   the features and confirm the count drops to 1.

**Previous:** [Episode 14](../14-kv-cache/) ·
**Next:** [Episode 16 — GemmaModel & RMS normalization](../16-gemma-model-and-rms-norm/)
