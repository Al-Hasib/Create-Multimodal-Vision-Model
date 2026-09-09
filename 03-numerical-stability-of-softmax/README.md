# Episode 03 — Numerical Stability of the Softmax

> **Video chapter:** `00:16:50 – 00:23:00` (Numerical stability of the Softmax)
> **Target runtime:** 10–12 min · **Code:** [`code/softmax_stability.py`](code/softmax_stability.py)

## Why this gets its own episode

We are about to write `softmax` at least four times (SigLIP attention, Gemma attention, the
CLIP loss, top-p sampling). Every one of them is a place where a textbook-correct formula
produces `nan` on real inputs. This is also the cleanest example of a theme that runs
through the whole playlist: **the formula in the paper is not the formula in the code.**

## The failure

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

`exp` grows absurdly fast, and floating point has a largest representable value:

| dtype | max value | `exp(x)` overflows above |
|---|---|---|
| float16 | 65,504 | x ≈ 11.09 |
| float32 | 3.4e38 | x ≈ 88.7 |

**11.09.** Attention logits are dot products of vectors with hundreds of components; values
in the tens or hundreds are completely normal. So in float16 — the dtype everyone infers
in — the naive formula is not an edge case, it is the default case.

Both directions break:

- all logits large and positive → `inf / inf` → `nan`
- all logits large and negative → `0 / 0` → `nan`

Once one `nan` appears it propagates through every subsequent operation, and the loss goes
to `nan` with no indication of where it started.

## The fix: shift by the maximum

For any constant `c`:

```
exp(x_i - c) / Σ_j exp(x_j - c)  =  (exp(x_i)/exp(c)) / (Σ_j exp(x_j)/exp(c))  =  softmax(x)_i
```

The `exp(c)` cancels exactly, so **the softmax is invariant to adding a constant to its
input.** Choose `c = max(x)`:

- the largest exponent becomes `exp(0) = 1` → nothing can overflow
- at least one denominator term is exactly `1` → the denominator is never 0
- every other term is in `(0, 1]` → underflow to 0 is harmless

```python
def stable_softmax(x):
    maximum = x.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(x - maximum)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)
```

This is free: same function, same output (the script measures `max diff ≈ 1e-8` over a
thousand random rows), no overflow. `torch.softmax` does it internally — which is why we
call `nn.functional.softmax` in the model instead of writing the formula out.

## The same trick for `log(softmax)`: log-sum-exp

Losses need `log(softmax(x))`. Computing it as `torch.log(torch.softmax(x))` is bad twice
over: the softmax may underflow a small probability to exactly 0, and `log(0) = -inf`.

```
log(softmax(x)_i) = x_i - logsumexp(x)     where   logsumexp(x) = max(x) + log(Σ_j exp(x_j - max(x)))
```

Subtraction instead of a division of exponentials — no intermediate probability, nothing
to underflow. This is why you always pass **logits** to `cross_entropy`, never
probabilities: it uses `log_softmax` internally.

## Where it shows up in our code

```python
# modeling_siglip.py and modeling_gemma.py, both attentions:
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
```

Two deliberate choices in one line:

1. **`dtype=torch.float32`** — upcast before the softmax even when the model runs in
   bfloat16, then cast the result back. The softmax is where precision matters most and
   costs least.
2. **`dim=-1`** — normalize across the *keys* of each query. Getting this axis wrong is a
   silent bug: the shapes still work, the rows still sum to something, the model just
   learns nonsense.

You will also see `torch.finfo(dtype).min` used as the masking value in Episode 15 rather
than `-inf`. Reason: a row that is *entirely* masked gives `-inf - (-inf) = nan` with
infinities, but a harmless uniform distribution with the dtype minimum.

## The connection to Flash Attention

The shift needs `max(row)` and the denominator needs `sum(row)`, so a naive softmax reads
each row of the `[Seq_Len, Seq_Len]` score matrix twice — and that matrix, not the weights,
is what dominates attention's memory:

| seq_len | scores per head | fp32 |
|---|---|---|
| 256 | 65 K | 0.26 MB |
| 1,024 | 1.0 M | 4.2 MB |
| 8,192 | 67 M | 268 MB |

Flash Attention keeps the running max and running sum as it walks the row in blocks,
rescaling the accumulator when a new max appears. Same numbers, same stability trick, one
pass, and the full matrix never touches memory. Our implementation materialises it — which
is fine for learning, and is exactly what you would replace first in production.

## Run it

```bash
cd 03-numerical-stability-of-softmax/code
python softmax_stability.py
```

## Key takeaways

- `softmax(x) == softmax(x - c)`; use `c = max(x)` and overflow is impossible.
- float16 overflows `exp` above x ≈ 11 — this is a routine input, not an edge case.
- Use `log_softmax` / pass logits to losses; never `log(softmax(x))`.
- Upcast the softmax to float32 even in a bfloat16 model.
- Mask with `finfo.min`, not `-inf`.

## Exercises

1. Write `naive_softmax` and find the smallest logit value that produces `nan` in float16,
   then in float32. Compare with the table above.
2. `cross_entropy(logits, y)` vs `nll_loss(log(softmax(logits)), y)`: make the two disagree
   by choosing extreme logits.
3. Implement `logsumexp` yourself and check it against `torch.logsumexp` for
   `x = [1000., 1001.]`.
4. **Harder:** implement one-pass "online softmax" — walk a vector in blocks of 4, keeping
   a running max and running sum, rescaling when the max changes. Verify it matches
   `torch.softmax`. You have just written the core of Flash Attention.

## Further reading

- Milakov & Gimelshein, *Online normalizer calculation for softmax*,
  <https://arxiv.org/abs/1805.02867>
- Dao et al., *FlashAttention*, <https://arxiv.org/abs/2205.14135>

**Previous:** [Episode 02](../02-contrastive-learning-and-clip/) ·
**Next:** [Episode 04 — SigLIP: the sigmoid loss](../04-siglip-sigmoid-loss/)
