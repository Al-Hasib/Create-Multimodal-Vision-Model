# Episode 19 — Rotary Positional Embedding

> **Video chapter:** `04:56:00 – 05:23:40` (Rotary Positional Embedding)
> **Target runtime:** 28–30 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/rope_demo.py`](code/rope_demo.py)

The last piece of the model, and the most elegant. After this episode
`modeling_gemma.py` is complete.

## The problem with absolute positions

SigLIP adds a learned vector per slot (Episode 05). That is **absolute**: slot 5 has its own
trained vector, unrelated to slot 6, and slot 5000 was never trained at all. But what
attention actually needs is **distance** — "the previous word", "three tokens back", "same
sentence". Absolute encodings force the model to *infer* relative distance from pairs of
absolute markers.

## The idea

Do not add anything. **Rotate** `q` and `k` by an angle proportional to their position.

The dot product of two rotated vectors depends only on the *difference* of the rotation
angles:

```
(R_m q) · (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
```

Rotation matrices compose by adding angles, and `R_m^T = R_{-m}`. So an attention score
between position `m` and position `n` automatically depends on `n - m` and nothing else.
Relative positions, for free, with no parameters.

## The construction

Split `head_dim` into `head_dim/2` pairs. Pair `i` gets its own frequency:

```
theta_i = base^(-2i/head_dim),    i = 0, 1, ..., head_dim/2 - 1,    base = 10000
```

```python
inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
```

Each pair is a 2-D plane rotated at its own speed:

| pair | theta | full turn every |
|---|---|---|
| 0 | 1.0 | ~6 positions |
| 1 | 0.1 | ~63 positions |
| 2 | 0.01 | ~628 positions |
| … | … | … |

A **multi-resolution clock**: fast pairs distinguish "immediately before me", slow pairs
distinguish "somewhere earlier in the document". This is also why RoPE has zero parameters —
`inv_freq` is derived from `base` and `dim`, registered `persistent=False` because it is not
in the checkpoint.

Then multiply the frequencies by the positions and take cos/sin:

```python
inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
position_ids_expanded = position_ids[:, None, :].float()
with torch.autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)    # [B, Seq_Len, Head_Dim]
    cos, sin = emb.cos(), emb.sin()
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

- The matmul is `[B, Head_Dim/2, 1] @ [B, 1, Seq_Len]` — an outer product giving every
  (position, frequency) pair.
- **`autocast(enabled=False)`** forces float32: `position_ids` can be in the thousands, and
  `position * theta` in bfloat16 loses enough precision to visibly corrupt the angles.
- **`torch.cat((freqs, freqs))`** duplicates the table so both halves of a pair get the same
  angle. Which brings us to the layout.

## `rotate_half` and the HF layout

The 2-D rotation of a pair `(x1, x2)` by angle `a` is:

```
x1' = x1*cos(a) - x2*sin(a)
x2' = x1*sin(a) + x2*cos(a)
```

Written for a whole vector, that is `x * cos + rotate_half(x) * sin`, where:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]     # first half
    x2 = x[..., x.shape[-1] // 2 :]     # second half
    return torch.cat((-x2, x1), dim=-1)
```

```
x              = [1, 2, 3, 4, 5, 6, 7, 8]
rotate_half(x) = [-5, -6, -7, -8, 1, 2, 3, 4]
```

So the pairs are **(0,4), (1,5), (2,6), (3,7)** — dimension `i` paired with `i + dim/2`, not
with `i+1`. The original paper interleaves adjacent dimensions instead. The two formulations
are related by a permutation of the head dimension, and **the checkpoint's weights were
stored to match this one.** Do not "fix" it — that is precisely how a re-implementation
breaks while looking more correct than the original.

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)   # add the head axis
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

`unsqueeze(1)` gives `cos`/`sin` a size-1 head axis, so one table broadcasts over all
heads — which is why `q` (8 heads) and `k` (1 head) can share it.

## The properties, measured

The demo verifies each of these against our own code:

**1. Only the distance matters.**

| q at | k at | distance | q·k after RoPE |
|---|---|---|---|
| 5 | 3 | 2 | −2.278603 |
| 6 | 4 | 2 | −2.278603 |
| 100 | 98 | 2 | −2.278602 |
| 5 | 4 | 1 | −2.266718 |

Same distance → same score, at position 5 or at position 100. Nothing had to learn this.

**2. Norms are preserved.** `‖q‖ = ‖RoPE(q, 42)‖` exactly — it is a rotation.

**3. Long-term decay.** Careful here, because it is easy to state wrongly. For two
*independent* random vectors, rotating them changes nothing on average (the dot product of
isotropic noise is rotation invariant) — the demo confirms this first. The decay appears for
a query that **matches** its key, which is the case attention cares about:

| distance | mean q·k (q = k) |
|---|---|
| 0 | 256.4 |
| 4 | 196.3 |
| 64 | 122.9 |
| 1024 | 44.7 |
| 4096 | ≈ 0 |

At distance 0 every cosine is 1 and the pairs add up; as the distance grows the fast pairs
fall out of phase and cancel. A soft prior toward nearby tokens, not a mask — attention can
still learn to look far back.

**4. Translation invariance.** Feed the *same* tokens at positions `1,2,3` and at
`101,102,103` and the output is **bit-identical**. Shift every position by a constant and no
distance changes. A learned absolute table would have given three unrelated answers. This is
the payoff, stated as a testable property.

## Where it is applied — and where it is not

- **Q and K only, never V.** V is the payload being averaged, not part of the matching.
- **After the projections, before the cache update** (Episode 18): a cached key must already
  carry its position.
- **Inside every layer**, because unlike an additive input encoding there is nothing to
  inherit from the embedding step.

## `base` and context extension

`rope_theta` sets the slowest wavelength, capping the range of distinguishable distances.
For the real `head_dim=256`:

| base | slowest pair turns once every |
|---|---|
| 10,000 | ~58,000 positions |
| 100,000 | ~574,000 positions |
| 1,000,000 | ~5,640,000 positions |

Raising the base after training ("NTK-aware scaling", "rope scaling") is the standard way to
extend a context window — and it works *because* RoPE is a function of the position rather
than a learned table. There is no equivalent trick for SigLIP's absolute embeddings, which
is why each resolution needs its own checkpoint. Gemma keeps `rope_theta=10000` with
`max_position_embeddings=8192`.

## Run it

```bash
cd 19-rotary-positional-embedding/code
python rope_demo.py
```

## Key takeaways

- Rotate Q and K by `position * theta_i` per dimension pair; the score then depends only on
  the relative distance.
- `theta_i = base^(-2i/dim)` — a multi-resolution clock, zero parameters.
- HF pairs dimension `i` with `i + dim/2` (`rotate_half`); the checkpoint matches this.
- `cos`/`sin` computed in float32 under `autocast(enabled=False)`.
- Q and K only, never V. After projection, before the cache.
- Shifting all positions by a constant changes nothing — that is what "relative" means.
- **`modeling_gemma.py` is now complete.**

## Gotchas

- **Interleaved pairing** instead of the half-split → wrong rotation for the checkpoint's
  weights, plausible-looking output.
- **Forgetting `torch.cat((freqs, freqs))`** → `cos`/`sin` half the width of the head, and
  the two halves of each pair rotate by different angles.
- **Applying RoPE to V**, or applying it twice (once before and once after the cache).
- **Computing angles in bfloat16** → large positions lose precision.
- **0-based vs 1-based `position_ids`** (Episode 15) → every distance is right but every
  absolute phase is shifted; harmless in theory, a mismatch against training in practice.
- **`unsqueeze_dim` wrong** → the broadcast silently applies one position's angles to the
  wrong axis.

## Exercises

1. Verify translation invariance yourself: pick random `q`, `k` and confirm the score at
   `(m, n)` equals the score at `(m+1000, n+1000)` to float precision.
2. Implement the paper's *interleaved* variant, then find the permutation matrix `P` such
   that `P` applied to the head dimension converts between the two. Confirm the scores match
   after permuting.
3. Set `base=1.0` and explain what happens to every frequency. Then set `base=1e12`.
4. Raise `rope_theta` to 1e6 with the real weights (Episode 20) and see how the captions
   change. Why does naive base scaling degrade short-context quality?
5. Apply RoPE to `value_states` as well and describe the failure mode in words before
   running it.
6. RoPE is applied in all 18 layers. Would applying it only in layer 0 work? Why not?

## Further reading

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*,
  <https://arxiv.org/abs/2104.09864>
- Biderman et al., *Rotary Embeddings: A Relative Revolution*,
  <https://blog.eleuther.ai/rotary-embeddings/>
- Peng et al., *YaRN: Efficient Context Window Extension*, <https://arxiv.org/abs/2309.00071>

**Previous:** [Episode 18](../18-grouped-query-attention/) ·
**Next:** [Episode 20 — Inference & top-p sampling](../20-inference-and-top-p-sampling/)
