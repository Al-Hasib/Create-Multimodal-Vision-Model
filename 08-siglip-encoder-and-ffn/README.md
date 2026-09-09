# Episode 08 — Coding SigLIP: the encoder layer and the FFN

> **Video chapters:** `01:05:28 – 01:20:45` (Coding SigLip (Encoder) · Coding SigLip (FFN))
> **Target runtime:** 15–18 min · **Code:** [`code/modeling_siglip.py`](code/modeling_siglip.py), [`code/check_encoder.py`](code/check_encoder.py)

We write everything around attention: the MLP, the encoder layer, the encoder stack, and
the two top-level wrappers. Attention itself is a **placeholder** in this episode —
deliberately. By the end the tower runs end to end, and the one thing it cannot do is mix
information between patches. That gap is the best possible motivation for Episodes 09–10.

## `SiglipMLP` — the feed-forward network

```python
class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        # [B, Num_Patches, Embed_Dim] -> [B, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [B, Num_Patches, Intermediate_Size] -> [B, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states
```

Up 4×, non-linearity, back down. Two observations:

- **It is applied per token.** `nn.Linear` acts on the last dimension only, so patch 5's
  output depends on patch 5's input and nothing else. The FFN is where knowledge is stored;
  attention is what moves information between positions. Keeping that division straight
  makes the whole architecture legible.
- **`approximate="tanh"`** is not a detail you get to choose. GELU's exact form needs
  `erf()`; the tanh form is a cheap approximation, and it is what the checkpoint was
  trained with. Mismatch it and outputs drift.

Why 4×? Empirical convention from the original transformer, kept ever since. The real
SigLIP-So400m uses 4304/1152 ≈ 3.7 — its shape came from a scaling study, not a round
number.

## `SiglipEncoderLayer` — pre-norm, two residual branches

```python
def forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, _ = self.self_attn(hidden_states=hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

Read it as: *"normalize a copy, transform it, add the result back to the untouched
original."* Twice — once for attention, once for the MLP.

The normalization sits **inside** the branch, never on the residual path (Episode 07). Both
`layer_norm1` and `layer_norm2` are `nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)`
with `eps=1e-6` from the config.

Note `self.self_attn` returns a tuple `(output, weights)` and we discard the weights with
`, _`. The signature exists so you can inspect attention maps; the model never needs them.

## `SiglipEncoder` — just a stack

```python
self.layers = nn.ModuleList(
    [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
)
...
for encoder_layer in self.layers:
    hidden_states = encoder_layer(hidden_states)
```

Shape in equals shape out for every layer, which is the property that lets you stack 27 of
them and choose the depth from a config file. `nn.ModuleList` (not a plain Python list) is
what registers the sublayers so their parameters appear in `parameters()` and `state_dict()`.

No mask is threaded through, because there is nothing to mask (Episode 05).

## `SiglipVisionTransformer` and `SiglipVisionModel`

```python
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        return self.post_layernorm(last_hidden_state)


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        return self.vision_model(pixel_values=pixel_values)
```

`post_layernorm` is needed because the last layer's residual add leaves the output
un-normalized — with pre-norm, someone has to normalize at the end.

**Why the two nested classes?** `SiglipVisionModel` adds nothing but a name. It exists so
that our parameter names match the checkpoint's: the weights are stored under
`vision_tower.vision_model.encoder.layers.0...`, and `load_state_dict` matches **by string
key**. Renaming or flattening a wrapper silently loads nothing (with `strict=False`, which
is what `utils.load_hf_model` uses) and the model outputs noise. This is the single most
common reason a from-scratch re-implementation "works" but produces garbage.

**No pooling head.** The output is `[Batch_Size, Num_Patches, Embed_Dim]` — all 256
embeddings, which is exactly what PaliGemma wants (Episode 13).

## The placeholder

This episode's `modeling_siglip.py` has:

```python
class SiglipAttention(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 10 (Multi-Head Attention)."""
    def forward(self, hidden_states):
        return hidden_states, None
```

So the tower is currently "LayerNorm + MLP, applied independently to each patch". Section 4
of the check script proves it: perturb patch 0 and **every other output token moves by
exactly 0.0000**. Re-run the same experiment in Episode 10 and all of them move. That
difference *is* attention.

## Run it

```bash
cd 08-siglip-encoder-and-ffn/code
python check_encoder.py
```

## Key takeaways

- The FFN is per token: expand 4×, GELU (tanh approximation), project back.
- Attention mixes positions; the MLP stores knowledge. Nothing else mixes positions.
- Pre-norm layer: `residual + sublayer(norm(x))`, twice per layer.
- Shape-preserving layers are what make depth a config parameter.
- `post_layernorm` cleans up after the final residual add.
- **Module names are part of the checkpoint contract** — keep the nesting.

## Gotchas

- **A plain `[...]` instead of `nn.ModuleList`** → the layers' parameters are invisible to
  `.parameters()`, `.to(device)` and `state_dict()`. The model trains nothing and loads
  nothing, with no error.
- **Renaming `vision_model`, `self_attn`, `layer_norm1`, `fc1`** → keys stop matching and
  `strict=False` swallows it silently.
- **Post-norm by accident** (`layer_norm(residual + x)`) → runs fine, trains badly, and
  loads a pre-norm checkpoint incorrectly.
- **Forgetting `residual = hidden_states` before the second branch** → the MLP's residual
  is the pre-attention value and you have quietly deleted the attention output.

## Exercises

1. Verify with `named_parameters()` that your key names match
   `vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight` once the model is
   nested inside PaliGemma (Episode 12).
2. Swap `nn.ModuleList` for a Python list and print `len(list(model.parameters()))` before
   and after. This is worth doing once so you recognise the failure later.
3. Replace `gelu(approximate="tanh")` with exact GELU and measure the max output difference
   over 27 layers. Does a "tiny" per-layer difference stay tiny?
4. Rewrite the layer in post-norm form, then compare gradient norms at layer 0 for both
   variants at depth 27.

**Previous:** [Episode 07](../07-batch-norm-vs-layer-norm/) ·
**Next:** [Episode 09 — Multi-head attention: theory](../09-multi-head-attention-theory/)
