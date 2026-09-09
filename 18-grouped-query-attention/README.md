# Episode 18 — Grouped-Query Attention and the Gemma attention block

> **Video chapters:** `04:16:02 – 04:56:00` (Multi-Head Attention coding · Grouped Query Attention `04:18:30` · KV-Cache coding `04:43:26`)
> **Target runtime:** 25–30 min · **Code:** [`code/modeling_gemma.py`](code/modeling_gemma.py), [`code/gqa_demo.py`](code/gqa_demo.py)

The last big component. Everything from Episode 09 applies; what is new is that **K and V
have fewer heads than Q**, and that the cache is wired in.

## MHA → MQA → GQA

| | KV heads | Cache | Quality |
|---|---|---|---|
| **MHA** (multi-head) | = Q heads | largest | baseline |
| **MQA** (multi-query) | 1 | smallest | slight loss |
| **GQA** (grouped-query) | `g`, between 1 and Q | tunable | close to MHA |

Only the **key and value** projections shrink. The number of query heads — and therefore the
number of distinct attention patterns — never changes. Each group of query heads shares one
K/V pair.

```
MHA (8 kv heads)          GQA g=2                      MQA (1 kv head)
q0 q1 q2 q3 q4 q5 q6 q7   q0 q1 q2 q3  q4 q5 q6 q7     q0 q1 q2 q3 q4 q5 q6 q7
k0 k1 k2 k3 k4 k5 k6 k7   \____k0____/  \____k1____/    \_________k0_________/
```

`paligemma-3b`'s text model is `num_attention_heads=8`, `num_key_value_heads=1` — pure MQA.

## Why: decoding is memory bound

To generate **one** token, a decoder reads every weight and the entire KV-cache from memory,
then does a trivial amount of arithmetic on it. Modern accelerators execute hundreds of
FLOPs in the time it takes to fetch a byte, so the *fetch* is the bottleneck (Episode 14).
Shrinking the cache directly buys tokens per second, and lets far more sequences fit in
memory at once.

For `paligemma-3b` at 2048 tokens, bfloat16:

| variant | kv heads | cache/sequence | k+v params/layer |
|---|---|---|---|
| MHA | 8 | 302 MB | 8,388,608 |
| GQA g=4 | 4 | 151 MB | 4,194,304 |
| GQA g=2 | 2 | 75 MB | 2,097,152 |
| **MQA (actual)** | **1** | **38 MB** | **1,048,576** |

Same 8 query heads in every row.

## `repeat_kv`

The attention matmul needs `Q` and `K` to have the same number of heads, so the narrow K/V
are expanded just before use:

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

The resulting layout is `[kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]` — **consecutive query
heads form a group.** The demo verifies exactly which output heads each input head lands in.

Get this wrong (interleaving `[kv0, kv1, kv0, kv1, ...]` instead) and the model runs, the
shapes are all valid, and every query is paired with the wrong key. There is no error, only
degraded output. Worth a unit test.

Note also what `repeat_kv` does *not* save: it materialises the expanded tensor, so the
attention matmul does the full MHA amount of work. **The saving is in the cache and the
weights, not in the FLOPs** — which is the right trade, because decoding is bandwidth bound.

## `GemmaAttention.__init__`

```python
self.num_heads = config.num_attention_heads              # 8
self.head_dim = config.head_dim                          # 256, from the config
self.num_key_value_heads = config.num_key_value_heads    # 1
self.num_key_value_groups = self.num_heads // self.num_key_value_heads   # 8

self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=config.attention_bias)
self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, max_position_embeddings=..., base=self.rope_theta)
```

Differences from `SiglipAttention` (Episode 10):

- `head_dim` comes from the **config**, not from `hidden_size // num_heads`.
- `k_proj` and `v_proj` are **narrower** than `q_proj` — the whole point.
- `bias=config.attention_bias` → `False`. Gemma has no attention biases.
- Each layer owns a `GemmaRotaryEmbedding` (Episode 19) and a `layer_idx` for the cache.

## `forward`, and the order that matters

```python
query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# 1. RoPE
cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# 2. then cache
if kv_cache is not None:
    key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

# 3. then expand
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)

attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
assert attention_mask is not None
attn_weights = attn_weights + attention_mask
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
attn_output = torch.matmul(attn_weights, value_states)

attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
attn_output = self.o_proj(attn_output)
```

**RoPE → cache → repeat.** Each arrow is load-bearing:

- **RoPE before the cache**, because a cached key must already carry its position. It is
  written once and re-read at every later step; rotating it later, or rotating it again,
  corrupts it.
- **Cache before `repeat_kv`**, because the cache must store the *narrow* tensors. Cache the
  expanded ones and you have thrown away the entire benefit of MQA while keeping all its
  code.

Note `self.rotary_emb(value_states, ...)` passes `value_states` only to read its dtype and
device — RoPE is never applied to V.

## The mask

```python
assert attention_mask is not None
attn_weights = attn_weights + attention_mask
```

**Additive**, pre-softmax, with `0` for "attend" and `torch.finfo(dtype).min` for "block"
(Episode 15 builds it, Episode 03 explains why not `-inf`). The `assert` is deliberate: the
mask carries the prefix-LM semantics, and silently defaulting it to `None` would produce a
model that works and is wrong.

## Run it

```bash
cd 18-grouped-query-attention/code
python gqa_demo.py
```

Section 6 runs five decode steps and prints the cache growing by one position each time
with `attn_weights` widening `[1,8,1,1] → [1,8,1,5]`. `key_cache[0]` keeps
`num_kv_heads=2` — the narrow shape.

RoPE is still stubbed (`cos=1, sin=0`), so the model is currently **position-blind**: it can
attend, but it cannot tell order. Episode 19 fixes exactly that.

## Key takeaways

- GQA shrinks only K and V heads; query heads and attention patterns are untouched.
- It is a memory-bandwidth optimisation, because decoding is bandwidth bound.
- `repeat_kv` groups **consecutive** query heads; interleaving is a silent disaster.
- `repeat_kv` saves cache and weights, not FLOPs.
- **RoPE → cache update → repeat_kv**, in that order.
- `head_dim` from the config; `bias=False`; `layer_idx` for the cache slot.
- The mask is additive and asserted to exist.

## Gotchas

- **Caching after `repeat_kv`** → 8× the memory, no benefit, no error.
- **RoPE after the cache update** → cached keys unrotated or double-rotated.
- **Interleaved `repeat_kv`** → wrong query/key pairing, plausible output.
- **`view(bsz, q_len, self.num_heads, ...)` for K and V** → wrong; they have
  `num_key_value_heads`. This one *does* usually raise, thankfully.
- **`math.sqrt(self.head_dim)` vs `hidden_size / num_heads`** → the same number for this
  checkpoint, different for others.
- **Applying RoPE to V.**

## Exercises

1. Write the unit test that catches interleaved `repeat_kv`: assert that output head `i`
   equals input head `i // n_rep`.
2. Move `repeat_kv` before `kv_cache.update` and print `key_cache[0].shape`. Confirm the
   memory blowup, and confirm nothing raises.
3. Set `num_key_value_heads=8` (full MHA) and compare `key_cache[0].numel()` with the
   MQA version at the same sequence length.
4. Which is faster on your machine for `q_len=1, kv_len=1024`: our `repeat_kv` + matmul, or
   `F.scaled_dot_product_attention` with `enable_gqa=True`? Explain the difference.
5. Set `num_attention_heads=6, num_key_value_heads=4` and read the error. Why must
   `num_heads % num_kv_heads == 0`?

## Further reading

- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints*, <https://arxiv.org/abs/2305.13245>
- Shazeer, *Fast Transformer Decoding* (MQA), <https://arxiv.org/abs/1911.02150>

**Previous:** [Episode 17](../17-decoder-layer-and-geglu-ffn/) ·
**Next:** [Episode 19 — Rotary positional embedding](../19-rotary-positional-embedding/)
