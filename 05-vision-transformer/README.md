# Episode 05 — The Vision Transformer

> **Video chapter:** `00:29:13 – 00:35:38` (Vision Transformer)
> **Target runtime:** 12–15 min · **Code:** [`code/vit_patch_embedding.py`](code/vit_patch_embedding.py)

## The whole idea

A transformer eats a sequence of vectors. An image is a grid of pixels. So: **cut the image
into fixed-size patches, flatten each patch, project it to `hidden_size`, and call the
result a sentence.** That is a Vision Transformer. No convolutional pyramid, no pooling
stages, no architecture designed for images at all.

```
 224x224x3 image            196 patches of 16x16x3          196 tokens of 768
┌───┬───┬───┬───┐
│   │   │   │   │           each patch: 16*16*3 = 768        ┌──────────┐
├───┼───┼───┼───┤    ──►    numbers, flattened        ──►    │ token 0  │
│   │   │   │   │           then one shared Linear           │ token 1  │
├───┼───┼───┼───┤                                            │   ...    │
│   │   │   │   │                                            │ token 195│
└───┴───┴───┴───┘                                            └──────────┘
```

Then run a standard transformer encoder over those tokens. The output is 196 (or 256)
**contextualized patch embeddings** — each vector still corresponds to its patch position,
but now carries information from the whole image.

## Why patches and not pixels

Attention is `O(seq_len²)`. A 224×224 image has 50,176 pixels → 2.5 billion attention
scores per head, per layer. At 16×16 patches the sequence is 196 → 38,416 scores. A 256×
reduction in sequence length, 65,536× in attention cost. Patch size is the knob that trades
spatial detail against compute, and it is why `paligemma-3b-pt-224` (patch 14, 256 tokens)
and `-pt-448` (patch 14, 1024 tokens) exist as separate checkpoints.

## Patch-and-project *is* a strided convolution

The literal implementation would be: `unfold` the image into patches, reshape, apply a
`Linear`. But a convolution whose **stride equals its kernel size** does exactly that: each
output position sees one non-overlapping `patch_size × patch_size` window, and the
`out_channels` filters are the rows of the projection matrix.

```python
self.patch_embedding = nn.Conv2d(
    in_channels=config.num_channels,   # 3
    out_channels=self.embed_dim,       # hidden_size
    kernel_size=self.patch_size,
    stride=self.patch_size,            # == kernel_size -> no overlap
    padding="valid",                   # no invented border pixels
)
```

The script proves the equivalence: `unfold + reshape + linear` and `Conv2d` with the same
weights agree to `1e-6`. One `Conv2d` call is faster and shorter, which is why every ViT
implementation writes it that way — but it is patch-and-project, not "a convnet".

Then two reshapes turn the grid into a sequence:

```python
patch_embeds = self.patch_embedding(pixel_values)  # [B, Embed_Dim, H/P, W/P]
embeddings = patch_embeds.flatten(2)               # [B, Embed_Dim, Num_Patches]
embeddings = embeddings.transpose(1, 2)            # [B, Num_Patches, Embed_Dim]
```

Patch order after `flatten(2)` is **row-major**: left-to-right, top-to-bottom. The model
never knows that; the position embeddings learn it.

## Positions: the transformer has no idea where a patch came from

Self-attention is permutation equivariant — shuffle the input tokens and you get the same
outputs in shuffled order. A patch from the top-left corner and one from the bottom-right
are indistinguishable to it. So we add position information explicitly:

```python
self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)),
                     persistent=False)
...
embeddings = embeddings + self.position_embedding(self.position_ids)
```

**Learned absolute** position embeddings: one trainable vector per slot, simply added.

- `nn.Embedding` here is just a lookup table of `num_patches` rows — a fancy way to index a
  parameter matrix.
- `register_buffer(..., persistent=False)`: `position_ids` is not a parameter (no gradient)
  but should move with `.to(device)`, and should not be saved in or loaded from the
  checkpoint. Both properties matter.
- The table has a **fixed size**. Feed a 448×448 image to a tower trained at 224×224 and
  you have more patches than rows — hence separate checkpoints per resolution rather than
  one flexible model.

Contrast this with the text side (Episode 19), which encodes positions by *rotating* Q and
K, relatively, with no parameters at all. The same model uses two completely different
schemes on its two halves — a nice illustration that positional encoding is a design
choice, not a law.

## No causal mask

A language model must not see the future. An image has no future: every patch may attend
to every other patch, in both directions. That is why our `SiglipAttention.forward` takes
no `attention_mask` argument at all, and why `SiglipEncoder.forward` has nothing to pass
down. It is the single biggest structural difference between the vision tower and the
language decoder.

## The configuration we are about to write

`SiglipVisionConfig`'s defaults are the familiar ViT-Base numbers (hidden 768,
intermediate 3072, 12 layers, 12 heads, patch 16). The real values come from
`config.json` in the checkpoint:

| | our defaults | `paligemma-3b-pt-224` (SigLIP-So400m) |
|---|---|---|
| `hidden_size` | 768 | 1152 |
| `intermediate_size` | 3072 | 4304 |
| `num_hidden_layers` | 12 | 27 |
| `num_attention_heads` | 12 | 16 |
| `patch_size` | 16 | 14 |
| `image_size` | 224 | 224 |
| tokens per image | 196 | **256** |
| parameters | ~86 M | ~412 M |

`head_dim = 1152 / 16 = 72`, and `4304 / 1152 ≈ 3.7` rather than exactly 4 — "So400m" is a
*shape-optimised* tower, sized by a scaling study rather than by round numbers.

## Run it

```bash
cd 05-vision-transformer/code
python vit_patch_embedding.py
```

## Key takeaways

- Image → patches → flatten → project → treat as a sequence of tokens.
- `Conv2d(kernel=patch, stride=patch, padding="valid")` **is** patch-and-project.
- Sequence length is `(image_size / patch_size)²`; patch size trades detail for compute.
- Attention is permutation equivariant, so positions must be added explicitly.
- SigLIP uses learned *absolute* positions, of fixed size — resolution is baked in.
- No causal mask anywhere in the vision tower.
- Output is one contextualized embedding per patch, with no pooling and no `[CLS]`.

## Exercises

1. Predict `num_patches` for `image_size=224, patch_size=32` before running it.
2. Set `image_size=225, patch_size=16`. `num_patches` is computed with integer division —
   does it still match the actual number of tokens the conv produces? What did
   `padding="valid"` do to the last row and column of pixels?
3. Remove the position embedding addition and verify empirically that shuffling the patches
   shuffles the outputs identically (permutation equivariance). Then put it back and show
   that it no longer holds.
4. Compute the parameter count of the patch embedding for patch 14 vs patch 32 at
   `hidden_size=1152`. Which dominates: the projection or the position table?

## Further reading

- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT),
  <https://arxiv.org/abs/2010.11929>
- Alabdulmohsin et al., *Getting ViT in Shape* (where "So400m" comes from),
  <https://arxiv.org/abs/2305.13035>

**Previous:** [Episode 04](../04-siglip-sigmoid-loss/) ·
**Next:** [Episode 06 — Coding SigLIP: config & embeddings](../06-coding-siglip-embeddings/)
