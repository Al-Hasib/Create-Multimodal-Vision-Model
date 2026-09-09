# Episode 17 — The Gemma decoder layer and the GeGLU FFN

> **Video chapters:** `04:09:50 – 04:16:02` (Gemma Decoder Layer · Gemma FFN (MLP) `04:12:44`)
> **Target runtime:** 10–12 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/check_decoder_layer.py`](code/check_decoder_layer.py)

A short episode: if you wrote the SigLIP encoder layer in Episode 08, you already know the
shape of this one. The interest is in what changed.

## `GemmaDecoderLayer`

```python
def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _, = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, kv_cache=kv_cache,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

Structurally **identical** to `SiglipEncoderLayer`: pre-norm, two residual branches, one for
attention and one for the MLP. All the differences are in the parts:

| | SigLIP encoder layer | Gemma decoder layer |
|---|---|---|
| Norm | `nn.LayerNorm` | `GemmaRMSNorm` |
| FFN | GELU, 2 matrices | **GeGLU, 3 matrices** |
| Attention | no mask, MHA, no positions | mask, **GQA**, **RoPE** |
| Extra args | none | `attention_mask`, `position_ids`, `kv_cache` |

That the two halves of a multimodal model share one layer template is worth saying out
loud — it is why "vision transformer" and "language transformer" are the same object with
different plumbing.

**A naming trap:** `post_attention_layernorm` normalizes the *input to the MLP*. Despite the
name it does not sit after the attention residual add. Same for `input_layernorm`, which is
the input to attention, not to the layer. Both names come from the checkpoint, so we keep
them.

Also note `hidden_states, _, = self.self_attn(...)` — the trailing comma is legal Python
tuple unpacking, discarding the attention weights.

## `GemmaMLP` — GeGLU

```python
class GemmaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
```

**Three** matrices instead of two, and an element-wise product:

```
            ┌── gate_proj ──► GELU ──┐
   x ───────┤                        ⊙ ───► down_proj ───► out
            └── up_proj ─────────────┘
```

A classic FFN is `down(gelu(up(x)))`: expand, threshold, contract. GeGLU splits the
expansion into two independent projections and lets one **modulate** the other. The gated
branch decides *how much of each channel to let through*; the other branch supplies the
content. Where a plain FFN can only apply a fixed non-linearity per channel, GeGLU learns a
per-token, per-channel gate.

The check script measures how many gate values are effectively zero for a given token —
those channels are switched off, and a different token switches off a different set. One
shared FFN, per-token behaviour.

Two practical notes:

- **`bias=False` on all three.** Gemma drops the FFN biases (and the attention biases, via
  `attention_bias=False`). Add them and the checkpoint keys mismatch.
- **50% more parameters** at the same `intermediate_size` — which is why models using GLU
  variants shrink `intermediate_size` to about `2/3` of what they would otherwise use, to
  keep the parameter count constant. Shazeer's *GLU Variants Improve Transformer* is the
  reference; the finding is that per-FLOP, gating wins.

For `paligemma-3b`: `hidden 2048 → intermediate 16384` is an 8× expansion, and
`3 × 2048 × 16384 = 100.7 M` parameters per layer — **11× the attention block**
(Episode 16's budget). Most of a modern LLM is feed-forward.

## Why `approximate="tanh"` again

Same reason as Episode 08: the exact GELU uses `erf()`, the tanh form is a cheap
approximation, and the checkpoint was trained with the approximation. The script prints both
side by side — the difference is ~1e-3 per activation, which compounds over 18 layers.

## Run it

```bash
cd 17-decoder-layer-and-geglu-ffn/code
python check_decoder_layer.py
```

`self_attn` is still the Episode 18 stub, so this checks the wiring, the GeGLU
decomposition, and the gate behaviour.

## Key takeaways

- The decoder layer is the SigLIP encoder layer with different parts: RMSNorm, GeGLU, and a
  masked, position-aware, grouped-query attention.
- `input_layernorm` / `post_attention_layernorm` both normalize *branch inputs*, despite the
  names.
- GeGLU = `down(gelu(gate(x)) * up(x))`: one branch gates the other.
- `bias=False` everywhere in Gemma's linear layers.
- The FFN holds ~11× more parameters per layer than attention.

## Gotchas

- **Applying GELU to `up_proj` instead of `gate_proj`** — runs, loads, subtly wrong. The
  gate is the one that gets the non-linearity.
- **`+` instead of `*`** between the branches — that is not a GLU any more.
- **Adding biases** → key mismatch against the checkpoint.
- **Forgetting the second `residual = hidden_states`** → the MLP's residual is the
  pre-attention value, silently deleting the attention output.
- **Assuming `post_attention_layernorm` runs after the residual add**, and reordering the
  code to "fix" it.

## Exercises

1. Count `GemmaMLP`'s parameters and compare with a 2-matrix FFN at the same
   `intermediate_size`. Then find the `intermediate_size` that equalises them (hint: 2/3).
2. Swap `gate_proj` and `up_proj` in the forward. Does it still load the checkpoint? Does it
   still work? Two different questions — answer both.
3. Replace GeGLU with `down(gelu(up(x)))` and measure the output difference with real
   weights. Which parameters are now unused?
4. With real weights, print `residual.norm()` and the branch output's norm at each of the 18
   layers. Which dominates, and what does that say about how much each layer changes the
   residual stream?

## Further reading

- Shazeer, *GLU Variants Improve Transformer*, <https://arxiv.org/abs/2002.05202>
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*,
  <https://arxiv.org/abs/1606.08415>

**Previous:** [Episode 16](../16-gemma-model-and-rms-norm/) ·
**Next:** [Episode 18 — Grouped-query attention](../18-grouped-query-attention/)
