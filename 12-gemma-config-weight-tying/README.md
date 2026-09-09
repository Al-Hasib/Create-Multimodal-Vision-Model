# Episode 12 — Gemma configs, weight tying and the PaliGemma skeleton

> **Video chapters:** `02:40:56 – 02:46:20` (Coding Gemma · Weight tying `02:43:44`)
> **Target runtime:** 15–18 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/check_config_and_tying.py`](code/check_config_and_tying.py)

Start `modeling_gemma.py`: two config classes, the projector, and the top-level model that
holds all three components.

## `GemmaConfig`

```python
class GemmaConfig():
    def __init__(
        self,
        vocab_size, hidden_size, intermediate_size,
        num_hidden_layers, num_attention_heads,
        num_key_value_heads,          # <- fewer K/V heads than Q heads (Episode 18)
        head_dim=256,                 # <- NOT hidden_size / num_heads
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,           # <- RoPE base (Episode 19)
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
```

Note the first six have **no defaults** — they must come from the checkpoint. Compare with
`SiglipVisionConfig`, where everything defaults to ViT-Base. Different philosophy, and the
missing defaults are a useful guard against silently building the wrong-sized model.

Three entries are forward references to later episodes: `num_key_value_heads` (18),
`rope_theta` (19), `rms_norm_eps` (16).

## `head_dim` is its own parameter

For `paligemma-3b-pt-224`'s text model:

```
num_attention_heads  8       (query heads)
num_key_value_heads  1       (K/V heads -> multi-query attention)
head_dim           256
hidden_size       2048
```

`8 × 256 = 2048`, so here `head_dim` *happens* to equal `hidden_size / num_heads` — but the
code never assumes it, and other Gemma variants break the coincidence. The projections are:

```
q_proj: 2048 -> num_heads    x head_dim = 2048
k_proj: 2048 -> num_kv_heads x head_dim =  256      <- 8x narrower
v_proj: 2048 -> num_kv_heads x head_dim =  256
o_proj: 2048 -> 2048
```

Eight query heads sharing one key/value head means an 8× smaller KV-cache. Episode 18.

## `PaliGemmaConfig`

The bridge. It receives the parsed `config.json` and builds both sub-configs:

```python
self.vision_config = SiglipVisionConfig(**vision_config)
self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
self.vocab_size = self.text_config.vocab_size

self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
self.vision_config.projection_dim = projection_dim
```

The last two lines are the interesting ones — **derived**, not read from the file:

- `num_image_tokens = (224 / 14)² = 256`. Computed from the vision geometry, then handed to
  the processor so it emits exactly 256 `<image>` placeholders. One source of truth.
- `projection_dim = 2048` tells the projector its output width, which must equal the text
  model's `hidden_size`. This is the seam between the two models:

```
vision hidden 1152  ──Linear──►  projection_dim 2048  ==  text hidden 2048
```

`image_token_index=256000` places `<image>` just above the 256,000-token text vocabulary,
with the loc/seg tokens filling the rest up to 257,152.

## `PaliGemmaMultiModalProjector`

The entire multimodal bridge:

```python
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size,
                                config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [B, Num_Patches, Embed_Dim] -> [B, Num_Patches, Projection_Dim]
        return self.linear(image_features)
```

One `nn.Linear`. No activation, no norm, no depth. It works because the SigLIP embeddings
are *already* language-aligned (Episode 04) — the projector only has to change the
dimensionality, not learn a new semantics. Other VLMs use a 2-layer MLP or a resampler; 
PaliGemma's simplicity is the payoff of choosing a contrastive vision encoder.

## The skeleton

```python
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = GemmaForCausalLM(config.text_config)
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
```

Those three attribute names are the top level of every key in the checkpoint
(`vision_tower.vision_model...`, `language_model.model.layers...`). They are API, not style.

## Weight tying

```python
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
```

Both matrices are `[vocab_size, hidden_size]`. `embed_tokens` maps a token id → a vector;
`lm_head` maps a vector → a score per token. They are inverse operations, so sharing one
matrix is a sensible inductive bias — and with a tied head, "the score of token *t*" is
literally "how aligned is my hidden state with the embedding of token *t*" (the check script
verifies `lm_head(h) == h @ embed_tokens.weight.T`).

It is also a lot of memory:

```
257,152 x 2,048 = 526,647,296 parameters = 1.05 GB in bfloat16, not stored twice
```

That is ~21% of the text model — the price of a 257k vocabulary.

**One line of assignment, three consequences.** The checkpoint contains no
`lm_head.weight` entry at all, which is why `utils.load_hf_model` does:

```python
model.load_state_dict(tensors, strict=False)   # tolerate the missing lm_head key
model.tie_weights()                            # then create it by aliasing
```

Order matters: tying *before* loading would work too (loading `embed_tokens` writes through
the alias), but tying after is the safe habit — `load_state_dict` can replace tensors and
break the aliasing. If you forget `tie_weights()` entirely, `lm_head` keeps its random
initialisation and the model emits pure noise while loading "successfully".

## Run it

```bash
cd 12-gemma-config-weight-tying/code
python check_config_and_tying.py
```

Watch `same object? False` become `True`, then a write to `embed_tokens` appear in
`lm_head`.

## Key takeaways

- `GemmaConfig`'s core fields have no defaults — the checkpoint must supply them.
- `head_dim` is independent of `hidden_size / num_heads`; K/V get fewer heads than Q.
- `PaliGemmaConfig` *derives* `num_image_tokens` and `projection_dim`, wiring the two models
  together.
- The multimodal bridge is a single `nn.Linear` with a bias.
- `tie_weights()` aliases `lm_head.weight` to `embed_tokens.weight`: better prior, 1 GB
  saved, and mandatory because the checkpoint omits `lm_head`.
- Attribute names must match the checkpoint keys.

## Gotchas

- **`self.lm_head.weight = self.model.embed_tokens.weight`** — assign the `.weight`
  attribute (a `Parameter`), not `.weight.data`, and not the module.
- **Skipping `tie_weights()`** → a random LM head, garbage output, no error.
- **`strict=True` in `load_state_dict`** → fails on the missing `lm_head.weight`.
- **Renaming `vision_tower` / `language_model` / `multi_modal_projector`** → nothing loads,
  silently, thanks to `strict=False`.
- **Assuming `head_dim == hidden_size // num_heads`** works for this checkpoint and breaks
  on the next one.

## Exercises

1. Print `model.state_dict().keys()` and compare against the real checkpoint's keys
   (`safe_open(...).keys()`). Any mismatch is a weight that will not load.
2. Load the model, skip `tie_weights()`, and print `lm_head.weight.std()` vs
   `embed_tokens.weight.std()`. Which one is the untrained one?
3. Compute the total parameter count from the config by hand (embedding + 18 × (attention +
   MLP)) and check it against `sum(p.numel() for p in model.parameters())` — remembering
   that tying means *not* double counting.
4. Replace the projector with a 2-layer MLP. It will no longer load the checkpoint — which
   key fails, and what would you have to train?

**Previous:** [Episode 11](../11-paligemma-input-processor/) ·
**Next:** [Episode 13 — Merging image & text embeddings](../13-merging-image-and-text-embeddings/)
