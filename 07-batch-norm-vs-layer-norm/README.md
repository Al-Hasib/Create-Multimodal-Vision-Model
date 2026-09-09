# Episode 07 — Batch Normalization vs Layer Normalization

> **Video chapter:** `00:54:25 – 01:05:28` (Batch Normalization, Layer Normalization)
> **Target runtime:** 12–15 min · **Code:** [`code/norm_compare.py`](code/norm_compare.py)

## The problem both of them solve

Stack 27 layers and activations drift. If a layer's inputs grow, its outputs grow more, and
by layer 20 you are in a region where the activation function saturates (gradient ≈ 0) or
the values overflow. Worse, the *distribution* of each layer's input keeps moving as the
layers below it update — so every layer is chasing a moving target and has to keep
re-learning. (The classic name for this is "internal covariate shift"; the mechanism is
debated, the practical benefit of normalizing is not.)

The fix in both cases: **rescale activations to a known range, then let the model learn to
scale and shift them back if it wants to.** The only question is *which numbers you average
over.*

## The difference, in one picture

Activations are `[Batch_Size, Seq_Len, Features]`:

```
BatchNorm: one mean/var PER FEATURE,        LayerNorm: one mean/var PER TOKEN,
           computed down the batch                     computed across features

   features ──────────►                        features ──────────►
  ┌───┬───┬───┬───┐                           ┌───┬───┬───┬───┐
  │ ▓ │ ░ │ ▒ │ █ │  sample 0                 │ ▓ ▓ ▓ ▓ │        sample 0  ◄── one stat
  ├───┼───┼───┼───┤                           ├───┼───┼───┼───┤
  │ ▓ │ ░ │ ▒ │ █ │  sample 1                 │ ░ ░ ░ ░ │        sample 1  ◄── one stat
  ├───┼───┼───┼───┤                           ├───┼───┼───┼───┤
  │ ▓ │ ░ │ ▒ │ █ │  sample 2                 │ ▒ ▒ ▒ ▒ │        sample 2  ◄── one stat
  └───┴───┴───┴───┘                           └───┴───┴───┴───┘
    ▲ one stat per column                       each row normalized on its own
```

Formulas:

```
BatchNorm: y = (x - mean_batch(x))  / sqrt(var_batch(x) + eps)  * gamma + beta
LayerNorm: y = (x - mean_features(x)) / sqrt(var_features(x) + eps) * gamma + beta
```

Identical algebra. Different axis. Everything below follows from that axis.

## Why transformers use LayerNorm

**BatchNorm couples your samples.** The statistics of feature `j` depend on every other
sample in the batch, so sample 0's output changes when sample 3 changes. The script
demonstrates exactly this: perturb sample 3, watch sample 0's BatchNorm output move while
its LayerNorm output stays bit-identical. Consequences:

1. **Small batches → noisy statistics.** With batch size 1 the per-position variance is
   *zero*, and BatchNorm has nothing to work with. Autoregressive generation is batch
   size 1 with one token at a time — the worst case imaginable.
2. **Train ≠ eval.** BatchNorm needs running averages collected during training and a
   `model.train()` / `model.eval()` switch. Forgetting `.eval()` at inference is one of the
   most common PyTorch bugs, and it exists *because* of BatchNorm.
3. **Variable-length sequences.** Padding tokens land in the per-feature statistics and
   pollute them.
4. **Distributed training** needs the batch statistics synchronised across devices
   (`SyncBatchNorm`), which is a communication cost per norm layer.

LayerNorm has none of these: it is a pure function of a single token's own features. Batch
size 1 works, train and eval are identical, padding is irrelevant, no communication.

There is also a geometric difference worth noticing in the script's output: LayerNorm
re-centers *and* rescales, so it moves the vector towards the origin; RMSNorm (Episode 16)
only rescales, preserving direction exactly.

## `gamma` and `beta`: normalization is not a straitjacket

```python
layer_norm = nn.LayerNorm(features)
layer_norm.weight   # gamma, initialised to ones
layer_norm.bias     # beta,  initialised to zeros
```

Zero mean and unit variance is the *starting point*, not the requirement. The two learnable
per-feature vectors let the model recover any scale and offset it needs — including undoing
the normalization entirely if that is optimal. Without them, normalization would be a hard
constraint on what each layer can express.

## Pre-norm vs post-norm

Where the norm goes matters as much as which norm it is. Our SigLIP encoder layer
(Episode 08) is **pre-norm**:

```python
residual = hidden_states
hidden_states = self.layer_norm1(hidden_states)     # norm INSIDE the branch
hidden_states, _ = self.self_attn(hidden_states)
hidden_states = residual + hidden_states            # clean residual path
```

The original 2017 transformer was **post-norm** (`layer_norm(residual + sublayer(x))`),
which puts a normalization on the residual highway itself and rescales the gradient at
every layer. Pre-norm leaves an unobstructed path from the loss to layer 0, which is why
essentially every modern deep transformer — including both halves of PaliGemma — uses it,
and why they train without the learning-rate warmup that post-norm needs.

## Where each one appears in our model

| Component | Normalization | Episode |
|---|---|---|
| SigLIP encoder layer | `nn.LayerNorm`, `eps=1e-6`, pre-norm, ×2 per layer | 08 |
| SigLIP after the encoder | `post_layernorm` (`nn.LayerNorm`) | 08 |
| Gemma decoder layer | `GemmaRMSNorm`, pre-norm, ×2 per layer | 16–17 |
| Gemma after the stack | `GemmaRMSNorm` | 16 |

RMSNorm is LayerNorm with the mean subtraction and the bias removed. Same axis, same idea,
fewer operations — see Episode 16 for why that turns out to be enough.

## Run it

```bash
cd 07-batch-norm-vs-layer-norm/code
python norm_compare.py
```

## Key takeaways

- Same formula, different axis: BatchNorm averages over the batch, LayerNorm over features.
- BatchNorm makes one sample's output depend on the others → breaks at batch size 1,
  needs running stats and a train/eval switch, hates padding, needs syncing.
- LayerNorm is per token, so none of that applies.
- `gamma`/`beta` keep normalization from being a constraint.
- Pre-norm keeps the residual path clean; that is why modern transformers use it.

## Gotchas

- **`nn.LayerNorm(dim)` normalizes the *last* dimension** by default. Pass a tuple to
  normalize more.
- **`unbiased=False`** — normalization layers use the biased (population) variance. If you
  reimplement it with `torch.var` and leave the default `unbiased=True`, you get a slightly
  different answer and a confusing mismatch.
- **`eps` goes inside the square root**, and its value is part of the checkpoint's contract:
  SigLIP uses `1e-6`, and a different `eps` gives measurably different outputs.
- **`nn.BatchNorm1d` expects `[B, C, L]`**, not `[B, L, C]` — the axis convention that
  makes it awkward for sequences in the first place.

## Exercises

1. Reimplement LayerNorm from scratch and match `nn.LayerNorm` to `1e-6`. Then break it by
   using `unbiased=True` and measure the difference.
2. Run BatchNorm1d on a batch of 1 in `train()` mode and explain the output.
3. Take the Episode 08 encoder, build a 27-layer version, and compare the gradient norm at
   layer 0 for pre-norm vs post-norm on random data. This is the whole argument, measured.
4. LayerNorm needs `2 * features` extra parameters per layer. Compute the total for the
   real SigLIP tower (1152 wide, 27 layers, 2 norms per layer + 1). Is it worth removing?

## Further reading

- Ba et al., *Layer Normalization*, <https://arxiv.org/abs/1607.06450>
- Ioffe & Szegedy, *Batch Normalization*, <https://arxiv.org/abs/1502.03167>
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (pre- vs post-norm),
  <https://arxiv.org/abs/2002.04745>

**Previous:** [Episode 06](../06-coding-siglip-embeddings/) ·
**Next:** [Episode 08 — The SigLIP encoder & FFN](../08-siglip-encoder-and-ffn/)
