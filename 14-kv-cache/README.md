# Episode 14 — The KV-Cache

> **Video chapter:** `03:08:54 – 03:33:35` (KV-Cache (Explanation))
> **Target runtime:** 22–25 min · **Code:** [`code/kv_cache_demo.py`](code/kv_cache_demo.py)

Pure theory episode, and one of the highest-value ideas in the playlist: the difference
between a toy generation loop and a usable one.

## The redundancy

Generation is a loop, and the naive version re-runs the whole prefix every step:

```
step 1: forward([p1 p2 p3])           -> predict t1
step 2: forward([p1 p2 p3 t1])        -> predict t2      p1..p3 recomputed
step 3: forward([p1 p2 p3 t1 t2])     -> predict t3      p1..t1 recomputed
```

Now the key observation. In a causal model, token `i`'s row of the attention matrix depends
only on tokens `≤ i`. When token `i+1` arrives, **nothing about token `i` changes.** Its
key and value vectors were computed from its own embedding and its own position; they are
final the moment they are computed.

So: compute K and V once per position, keep them, and each step only needs a **query** for
the single new token.

## Prefill vs decode

The cache splits generation into two phases with completely different characteristics:

| | **Prefill** | **Decode** |
|---|---|---|
| When | the first forward | every step after |
| `q_len` | the whole prompt (261 for us) | **1** |
| Cache | written | appended to, and read entirely |
| Bound by | compute — big matmuls | **memory bandwidth** |
| Mask | needed (in general) | nothing to mask — there is no future |

Decode is the surprising one: to produce a single token you read every weight in the model
(2.9 B parameters) plus the entire cache, and then do a trivial amount of arithmetic on it.
Arithmetic intensity is terrible, the accelerator idles waiting on memory, and **the cache
size — not the FLOP count — is what limits how many users fit on a GPU.** That is the
context in which Episode 18's grouped-query attention exists.

## The saving

Per layer, generating `n` tokens after a prompt of length `p`:

- **no cache:** `Σ_t O((p+t)² · d)` — quadratic in the total length
- **cache:** `Σ_t O((p+t) · d)` — linear

Measured on the script's toy attention (`d = 128`, CPU):

| tokens | no cache | with cache | speedup |
|---|---|---|---|
| 128 | 20 ms | 8 ms | 2.5× |
| 512 | 270 ms | 39 ms | 7× |
| 1024 | 1368 ms | 81 ms | **17×** |

(Your absolute numbers will differ; the trend is the point.)

The gap widens with length, as the asymptotics predict. The script also verifies the
outputs are **identical** — this is a pure optimisation, not an approximation.

## The cost: memory

```
cache bytes = 2 (K and V) x layers x kv_heads x head_dim x seq_len x bytes_per_element
```

For `paligemma-3b-pt-224` (18 layers, `head_dim` 256, bfloat16) at 1024 tokens:

| | cache per sequence |
|---|---|
| with 8 KV heads (plain MHA) | 151 MB |
| **with 1 KV head (actual)** | **18.9 MB** |

Multiply by batch size. For a 70 B model — 80 layers, 8 KV heads, 4k context — you are into
gigabytes *per user*, which is why every serving system has a KV-cache eviction strategy and
why paged attention (vLLM) exists.

## What we will write (Episode 15 onwards)

```python
class KVCache():
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []     # one entry per layer
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]          # [B, Num_Heads_KV, Seq_Len, Head_Dim]

    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)       # first time for this layer
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

Design points:

- **`dim=-2` is the sequence dimension.** With `[B, Num_Heads_KV, Seq_Len, Head_Dim]` that
  is the axis to grow. Using `dim=-1` concatenates along `head_dim` and produces a tensor
  that "works" for a while — a genuinely nasty bug.
- **One list entry per layer**, indexed by `layer_idx`. This is why `GemmaAttention.__init__`
  takes and stores `layer_idx`, and why `GemmaDecoderLayer` passes it down.
- **`num_items()` reads layer 0** and doubles as the phase detector: `0` means prefill,
  anything else means decode (Episode 15 branches on it).
- **The cache lives outside the model.** `inference.py` creates one `KVCache`, passes it into
  every forward, and gets it back in the output dict. The model stays stateless, which keeps
  it easy to reason about.
- **`torch.cat` reallocates** the whole cache every step — O(n²) copying over a generation.
  Real serving systems preallocate to `max_length` and write into a slice. Good exercise.

## Run it

```bash
cd 14-kv-cache/code
python kv_cache_demo.py
```

## Key takeaways

- K and V for a position never change once computed → cache them, query only the new token.
- Prefill (compute bound, full prompt) and decode (memory bound, `q_len == 1`) are different
  workloads.
- Quadratic → linear; identical outputs, measured 17× at 1024 tokens.
- Cache size scales with `layers × kv_heads × head_dim × seq_len` and is what limits
  batch size in serving.
- Grow the cache along `dim=-2`, one entry per layer, `num_items()` distinguishes the phases.

## Gotchas

- **`dim=-1` instead of `dim=-2`** in the `cat`.
- **Reusing a `KVCache` between two prompts** — you get the previous conversation's keys.
  `inference.py` creates a fresh one per generation.
- **Forgetting to pass `layer_idx`** → every layer writes into cache slot 0 and reads back
  another layer's keys. Shapes are all valid. Output is garbage.
- **Caching K/V *before* applying RoPE** (Episode 19) → the cached keys carry no position,
  and the whole thing is subtly wrong. Order matters: project → RoPE → cache.
- **Caching the `repeat_kv`-expanded tensors** (Episode 18) → you store 8× more than you
  need and lose the entire benefit of MQA.

## Exercises

1. In the script, change the cache `cat` to `dim=-1` and find out how many steps it takes
   before something actually raises.
2. Implement a preallocated cache: `torch.zeros(B, H, max_len, D)` plus a write index.
   Compare timing against the `cat` version for 1024 tokens.
3. Compute the cache size for `paligemma-3b-pt-448` (1024 image tokens) with a 100-token
   answer, at batch size 32.
4. Implement a sliding-window cache that keeps only the last 256 positions. What breaks
   about the image tokens? (This is a real design problem in long-context VLMs.)
5. The vision tower runs on every decode step in `inference.py` even though the image never
   changes. Cache the image features and measure the speedup.

## Further reading

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*,
  <https://arxiv.org/abs/1911.02150>
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM),
  <https://arxiv.org/abs/2309.06180>

**Previous:** [Episode 13](../13-merging-image-and-text-embeddings/) ·
**Next:** [Episode 15 — Attention mask & position ids](../15-attention-mask-and-position-ids/)
