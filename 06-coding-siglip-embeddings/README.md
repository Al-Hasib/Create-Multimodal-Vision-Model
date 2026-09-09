# Episode 06 — Coding SigLIP: the config and the vision embeddings

> **Video chapter:** `00:35:38 – 00:54:25` (Coding SigLip)
> **Target runtime:** 18–20 min · **Code:** [`code/modeling_siglip.py`](code/modeling_siglip.py), [`code/check_embeddings.py`](code/check_embeddings.py)

**First code episode.** Create `modeling_siglip.py` and write two classes:
`SiglipVisionConfig` and `SiglipVisionEmbeddings`.

## Imports — that is the entire dependency list

```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
```

No `transformers`, no `timm`. Every layer from here to Episode 19 is built from `nn.Linear`,
`nn.Conv2d`, `nn.Embedding`, `nn.LayerNorm` and tensor operations.

## `SiglipVisionConfig`

A plain container — no inheritance, no validation, just named numbers with defaults:

```python
class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,          # Embed_Dim: the width of the model
        intermediate_size=3072,   # the FFN's inner width (4x hidden)
        num_hidden_layers=12,     # how many encoder layers
        num_attention_heads=12,
        num_channels=3,           # RGB
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs,
    ):
```

Three things to point out while typing it:

- **`**kwargs` is load-bearing.** We build this straight from the checkpoint's
  `config.json`, which contains keys we do not implement (`model_type`, `torch_dtype`, …).
  Without `**kwargs` the constructor raises on a real config file.
- **The defaults are ViT-Base, not PaliGemma.** The real tower is 1152/4304/27/16 with
  `patch_size=14`. Defaults are for testing; the checkpoint always overrides them.
- **`num_image_tokens`** defaults to `None` and is filled in later by `PaliGemmaConfig`
  (Episode 12). It exists so the processor knows how many `<image>` placeholders to emit.

`attention_dropout=0.0` — we only ever run inference, so dropout is a no-op, but the
parameter is kept so the code mirrors a trainable implementation.

## `SiglipVisionEmbeddings`

The whole class, then the commentary:

```python
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",           # no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)   # [B, Embed_Dim, H/P, W/P]
        embeddings = patch_embeds.flatten(2)                # [B, Embed_Dim, Num_Patches]
        embeddings = embeddings.transpose(1, 2)             # [B, Num_Patches, Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
```

**`stride == kernel_size`** is what makes the patches non-overlapping (Episode 05 proved
this is patch-and-project). **`padding="valid"`** means no pixel is invented at the border.

**`num_positions == num_patches`** — no `[CLS]` token. Classification ViTs prepend a
learnable token and read its output; SigLIP pools differently, and PaliGemma wants all the
patch embeddings anyway, so there is no extra slot.

**`register_buffer(..., persistent=False)`** — `position_ids` is `[[0, 1, 2, ..., N-1]]`.
It needs to follow the module across devices (so not a plain attribute), must not be
trained (so not a `Parameter`), and must not appear in `state_dict()` (so
`persistent=False`, otherwise `load_state_dict` on a checkpoint that lacks it complains).
The script verifies `'position_ids' in state_dict() == False`.

**The broadcast add.** `position_embedding(self.position_ids)` is
`[1, Num_Patches, Embed_Dim]` and the patch embeddings are `[Batch_Size, Num_Patches,
Embed_Dim]`. Broadcasting over the batch is intentional: every image in the batch gets the
same position vectors.

## Shape discipline

Every non-trivial line in this repository carries a shape comment:

```python
# [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
embeddings = embeddings.transpose(1, 2)
```

Keep doing this as you type. Nearly every bug in a from-scratch transformer is a shape or
axis bug that runs without error, and the comments are what let you spot the mismatch by
reading rather than by debugging.

## Run it

```bash
cd 06-coding-siglip-embeddings/code
python check_embeddings.py
```

Expected: 196 patches for the defaults, 256 for the So400m config, a
`[Embed_Dim, 3, P, P]` conv weight, and a demonstration that with a *constant* image every
token still differs — because the only thing distinguishing them is the position embedding.

## Key takeaways

- The config is a plain object; `**kwargs` lets it swallow a real `config.json`.
- `nn.Conv2d(kernel=patch, stride=patch, padding="valid")` patchifies and projects.
- `flatten(2).transpose(1, 2)` turns the `[B, C, H, W]` grid into a `[B, Seq, C]` sequence.
- Positions are a learned lookup table, added by broadcast, sized `num_patches`.
- `register_buffer(persistent=False)` for tensors that are neither parameters nor
  checkpoint contents.

## Gotchas

- **Forgetting `**kwargs`** → `TypeError` the moment you load a real config.
- **`flatten(1)` instead of `flatten(2)`** collapses the channel dimension into the spatial
  ones. Shapes still "work" downstream and the model silently produces garbage.
- **Adding positions before the transpose** broadcasts against the wrong axis.
- **`persistent=True` (the default)** on `position_ids` makes `load_state_dict(strict=True)`
  fail against the official checkpoint.

## Exercises

1. `patch_size=32`: predict `num_patches`, then check.
2. Print `patch_embedding.weight.shape` and explain each of the four dimensions in terms of
   "one linear layer applied per patch".
3. Replace the `Conv2d` with an explicit `unfold` + `Linear` that produces identical
   output (Episode 05's script has the recipe). Which is more readable? Which is faster?
4. Build the config for `paligemma-3b-pt-448` (`image_size=448`, `patch_size=14`). How many
   tokens? Why can that checkpoint not reuse the 224 model's position table?

**Previous:** [Episode 05](../05-vision-transformer/) ·
**Next:** [Episode 07 — Batch norm vs layer norm](../07-batch-norm-vs-layer-norm/)
