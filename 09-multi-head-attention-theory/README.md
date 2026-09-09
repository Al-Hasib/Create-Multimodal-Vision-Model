# Episode 09 — Multi-Head Attention: the theory

> **Video chapter:** `01:20:45 – ~01:50:00` (Multi-Head Attention, explanation half)
> **Target runtime:** 25–30 min · **Code:** [`code/attention_by_hand.py`](code/attention_by_hand.py)

The longest theory episode in the playlist, and the one worth re-watching. Episode 10 turns
all of this into 30 lines of code.

## The job

The MLP transforms each token on its own. Something has to let token `i` *look at* token
`j` and pull in what it needs. That is attention, and it is the only component in a
transformer that moves information between positions.

## Q, K, V

Three linear projections of the same input:

| | name | intuition |
|---|---|---|
| `Q = x @ W_q` | query | "what am I looking for?" |
| `K = x @ W_k` | key | "what do I have to offer?" |
| `V = x @ W_v` | value | "what I actually pass on if you pick me" |

The database analogy is apt: match each **query** against all **keys**, and retrieve a
weighted blend of the corresponding **values**. Unlike a database, the match is soft — you
get a bit of everything, weighted by relevance.

Note all three come from the *same* `x` here — that is what "self-attention" means. (In
cross-attention, Q comes from one sequence and K, V from another. PaliGemma does not use
cross-attention anywhere: the image is injected as *tokens*, not through a cross-attention
layer. That is a real architectural choice, and Episode 13 is where it happens.)

## The formula

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
```

Four steps:

1. **`Q @ K^T`** → `[Seq_Len, Seq_Len]`. Entry `(i, j)` = how much token `i` wants token `j`.
2. **`/ sqrt(d_k)`** → the scaling, see below.
3. **`softmax(..., dim=-1)`** → each *row* becomes a probability distribution over keys.
4. **`@ V`** → each output is a weighted average of value vectors.

### Why `sqrt(d_k)`

If `q` and `k` have independent unit-variance components, `q · k` has variance `d_k`. So
scores grow with the head dimension, push the softmax into saturation, and the gradient
vanishes. Dividing by `sqrt(d_k)` restores unit variance. The script measures it:

| `head_dim` | `std(q·k)` | after scaling |
|---|---|---|
| 8 | ≈ 2.8 | 1.0 |
| 64 | ≈ 8.0 | 1.0 |
| 512 | ≈ 22.6 | 1.0 |

Our code writes it as a multiply, `self.scale = self.head_dim**-0.5`, which is the same
thing and cheaper.

### Why `dim=-1`

The softmax must normalize across the **keys** of each query, so that each query spends a
total attention budget of 1. Using `dim=-2` normalizes down the columns instead: the code
runs, the shapes are unchanged, the rows no longer sum to 1, and the model is subtly and
permanently wrong. Check it with `weights.sum(-1)`.

## Why *multi*-head

One softmax row is one distribution — it can attend to essentially one thing. But a token
usually needs several relations at once: its syntactic head, the entity it refers to, the
adjacent token. One distribution cannot express "50% this, and separately 50% that" without
blurring them into an average.

So split the embedding into `num_heads` slices of `head_dim = embed_dim / num_heads`, run
independent attentions on each slice, and concatenate:

```
embed_dim 768 = 12 heads x head_dim 64
```

**It is free.** `h` heads of size `d/h` cost the same FLOPs as one head of size `d`, since
attention cost is linear in the head dimension. You get multiple attention patterns for the
price of one.

Finally `W_o` (`out_proj`) mixes the concatenated heads. Without it the heads write into
disjoint slices of the output and never combine — the layer would be `h` independent
sub-models sharing a residual stream.

## The shape dance

This is the part to type out by hand until it is automatic:

```python
q = self.q_proj(x)                                          # [B, Seq, Embed_Dim]
q = q.view(B, Seq, num_heads, head_dim)                     # split the last dim
q = q.transpose(1, 2)                                       # [B, num_heads, Seq, head_dim]
# ... same for k, v
attn = softmax(q @ k.transpose(2, 3) * scale, dim=-1)       # [B, num_heads, Seq, Seq]
out = attn @ v                                              # [B, num_heads, Seq, head_dim]
out = out.transpose(1, 2).contiguous()                      # [B, Seq, num_heads, head_dim]
out = out.reshape(B, Seq, embed_dim)                        # concatenate the heads
out = self.out_proj(out)
```

**Why the transpose?** `matmul` batches over all leading dimensions and only operates on the
last two. Moving `num_heads` in front of `Seq_Len` makes every `(batch, head)` pair an
independent attention problem — no loop needed.

**Why `.contiguous()`?** `transpose` does not move data, it only changes strides. `view`
requires a contiguous buffer, so merging `num_heads` and `head_dim` back together needs a
real copy first. (`reshape` will insert one for you; `view` raises.)

The script prints every intermediate shape and cross-checks the result against
`F.scaled_dot_product_attention` to `1e-7`.

## Masks

A mask is **added to the scores before the softmax**, using a very negative value so that
`exp(...)` ≈ 0 removes the token:

```python
attn_weights = attn_weights + attention_mask   # 0 = attend, finfo.min = block
```

Three variants matter for us:

| Where | Mask |
|---|---|
| SigLIP (image) | none — every patch sees every patch |
| Gemma (generated text) | causal — token `i` may not see the future |
| PaliGemma prompt | **none over the prefix** (image + prompt) — Episode 15 |

Use `torch.finfo(dtype).min` rather than `-inf`: a fully masked row gives
`-inf - (-inf) = nan` with infinities, but a harmless uniform row with the dtype minimum.

## Complexity

`Q @ K^T` is `O(Seq_Len² · d)` in time *and* memory: the score matrix is materialised. That
quadratic term is why patch size matters (Episode 05), why long context is hard, and what
Flash Attention attacks (Episode 03).

## Run it

```bash
cd 09-multi-head-attention-theory/code
python attention_by_hand.py
```

Small tensors, all numbers printed. Read the score matrix and the weight matrix and confirm
that each row of the weights sums to 1.

## Key takeaways

- Q/K/V are three projections of the same input; attention is soft dictionary lookup.
- `softmax(Q K^T / sqrt(d_k)) V`, softmax over `dim=-1` (the keys).
- `sqrt(d_k)` keeps score variance at 1, keeping the softmax out of saturation.
- Multiple heads = multiple simultaneous attention patterns, at no extra cost.
- `view` + `transpose(1, 2)` makes heads a batch dimension; `W_o` recombines them.
- Masks are additive, pre-softmax, using `finfo.min`.
- Cost is `O(Seq_Len²)` in time and memory.

## Exercises

1. Remove the `sqrt(d_k)` scaling with `head_dim=512` and print the maximum attention
   weight. How close to one-hot does the softmax get?
2. Change the softmax to `dim=-2` and verify the rows no longer sum to 1. Then explain what
   the layer is now computing.
3. Implement single-head attention with an explicit Python loop over positions, and match
   the vectorised version. Then time both.
4. Take an 8-head layer and zero out `W_o`'s off-diagonal blocks so heads cannot mix.
   Argue what capability the layer just lost.
5. Verify permutation equivariance: shuffle the input tokens and confirm the outputs are the
   same set, shuffled the same way. (This is *why* Episode 05 needed position embeddings.)

## Further reading

- Vaswani et al., *Attention Is All You Need*, <https://arxiv.org/abs/1706.03762>
- Alammar, *The Illustrated Transformer*, <https://jalammar.github.io/illustrated-transformer/>

**Previous:** [Episode 08](../08-siglip-encoder-and-ffn/) ·
**Next:** [Episode 10 — Multi-head attention: code](../10-multi-head-attention-code/)
