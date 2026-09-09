# Episode 13 — Merging image and text embeddings

> **Video chapter:** `02:46:20 – 03:08:54` (Coding Gemma)
> **Target runtime:** 20–22 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/check_merge.py`](code/check_merge.py)

This is the episode where the model actually becomes multimodal. After
`_merge_input_ids_with_image_features` there is no image and no text — only one sequence of
vectors, and the language model cannot tell which is which.

## The setup

The processor gave us `input_ids` that look like:

```
[<image> x256] [<bos>] [this] [building] [is] [\n]
     ids 256000          2      ...              108
```

Embedding those ids gives `[1, 261, 2048]` — but the 256 image slots currently hold the
embedding of the *placeholder token* `256000`, which is meaningless. We must replace them
with the projected patch embeddings.

## Three masks partition the sequence

```python
text_mask  = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
image_mask =  input_ids == self.config.image_token_index
pad_mask   =  input_ids == self.pad_token_id
```

Exactly one is `True` at every position. `input_ids` is `[B, Seq]`, so each mask must be
expanded to the embedding dimension before it can select vectors:

```python
text_mask_expanded  = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
```

`expand` does not copy — it creates a view with a zero stride. Cheap.

## Then three writes, in this order

```python
final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, ...)

# 1. text
final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
# 2. image
final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
# 3. padding
final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)
```

**Why `masked_scatter` for the image and `torch.where` for the text?**

`torch.where(cond, a, b)` needs `a`, `b` and `cond` broadcastable to one shape. The text
embeddings are already `[B, 261, 2048]` — same shape, element-wise choice, `where` is
perfect. But `image_features` is `[B, 256, 2048]` while the destination is `[B, 261, 2048]`:
the 256 vectors have to be **spread over the 256 True positions** of the mask.
`masked_scatter` walks the mask in order and consumes the source tensor one element at a
time, which is exactly that operation.

It also *silently* requires that the number of `True` entries matches the number of source
elements. This is the hard dependency between `num_image_tokens` in the processor and
`(image_size / patch_size)²` in the vision tower (Episode 12 derives one from the other for
precisely this reason).

**Why zero out padding last?** Position `pad` got a real vector from the embedding table
(the embedding of id 0), and `text_mask` already excluded it, so it still holds the zeros
from initialisation — but writing the zeros explicitly makes the invariant obvious and
survives someone reordering the writes. Note this codebase asserts there is no padding
anyway (Episode 11), so in practice this line never fires.

## The `sqrt(hidden_size)` division

```python
scaled_image_features = image_features / (self.config.hidden_size**0.5)
```

The single most puzzling line in the file, and it makes sense only in combination with
Episode 16:

```python
# GemmaModel.forward
normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
hidden_states = hidden_states * normalizer
```

Gemma multiplies the **entire** embedding sequence by `sqrt(2048) ≈ 45.25` on the way in —
its convention for scaling token embeddings. The text embeddings were trained expecting
that scale-up. The projected image features were not. So we pre-divide, the model
multiplies, and the image features arrive at their intended magnitude.

Remove this line and the image tokens enter roughly 45× too loud, drowning the prompt.
Nothing crashes; the captions just become nonsense. Exactly the kind of bug that motivates
building the thing by hand.

## Where this leaves us

`_merge_input_ids_with_image_features` returns three things, and in this episode only the
first is real:

```python
return final_embedding, None, None      # attention mask and position ids: Episode 15
```

The embeddings are merged, but nothing has told the model what positions these tokens
occupy or who may attend to whom. `forward` still raises `NotImplementedError`.

## Run it

```bash
cd 13-merging-image-and-text-embeddings/code
python check_merge.py
```

A tiny config (4 image tokens, hidden size 8) so the whole vectors print. The script
identifies, position by position, whether the merged vector came from the image, the text,
or the zero fill.

## Key takeaways

- The image is injected by **overwriting embeddings at the `<image>` positions** — no
  cross-attention.
- `text_mask`, `image_mask`, `pad_mask` partition the sequence; expand them to
  `embed_dim` before use.
- `torch.where` for same-shaped tensors, `masked_scatter` to spread a shorter tensor over a
  mask.
- `masked_scatter` requires `#True == #source` — the reason the two token counts are derived
  from one place.
- Image features are pre-divided by `sqrt(hidden_size)` to cancel Gemma's input scale-up.

## Gotchas

- **Using `torch.where` for the image features** → a broadcast error, or worse, a wrong
  broadcast.
- **Forgetting `.unsqueeze(-1).expand(...)`** → mask shape `[B, Seq]` against data
  `[B, Seq, D]`.
- **Skipping the `sqrt(hidden_size)` division** → runs fine, output is nonsense.
- **Reordering the writes** so padding is zeroed before the image scatter → the scatter
  overwrites the zeros or mismatches the count.
- **`image_token_index` disagreeing with the processor's `<image>` id** → `image_mask` is
  all `False`, `masked_scatter` writes nothing, and the model captions a blank image while
  looking perfectly healthy.

## Exercises

1. Delete the `/ sqrt(hidden_size)` and compare `merged[0,0].norm()` before and after. Then
   (after Episode 20) compare the actual generated caption.
2. Make `num_image_tokens` 3 while the tower produces 4 and read the `masked_scatter` error.
   Then make it 5 — is the failure as loud?
3. Set `image_token_index` to an id that never appears. Confirm the model still runs and
   produces text. What is it captioning?
4. Rewrite the image write using `index_put_` or advanced indexing instead of
   `masked_scatter`, and confirm identical output. Which version reads better?
5. Extend the function to two images per prompt. What changes in the mask, in the scatter,
   and in the processor?

**Previous:** [Episode 12](../12-gemma-config-weight-tying/) ·
**Next:** [Episode 14 — The KV-cache](../14-kv-cache/)
