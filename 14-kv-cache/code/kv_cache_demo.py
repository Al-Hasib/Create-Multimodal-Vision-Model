"""Episode 14 -- the KV-cache, from the redundancy it removes to the memory it costs.

Run:  python kv_cache_demo.py
"""

import math
import time

import torch

torch.manual_seed(0)

D = 128  # head dimension for the toy model


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        q_len, kv_len = scores.shape[-2:]
        mask = torch.full((q_len, kv_len), float("-inf")).triu(diagonal=kv_len - q_len + 1)
        scores = scores + mask
    return torch.softmax(scores, dim=-1) @ v


def main() -> None:
    print("=" * 74)
    print("1. THE REDUNDANCY")
    print("=" * 74)
    print("  Generation is a loop. Without a cache, step t re-runs the whole prompt:")
    print("      step 1: [p1 p2 p3]        -> predict t1")
    print("      step 2: [p1 p2 p3 t1]     -> predict t2      p1..p3 recomputed")
    print("      step 3: [p1 p2 p3 t1 t2]  -> predict t3      p1..t1 recomputed")
    print("\n  But a causal model's row for token i never changes when tokens after i")
    print("  arrive: K and V for a position are computed once and stay valid forever.")
    print("  So keep them. Only the *new* token needs a query.")

    print()
    print("=" * 74)
    print("2. SAME OUTPUT, WITH AND WITHOUT THE CACHE")
    print("=" * 74)
    seq_len = 6
    x = torch.randn(seq_len, D)
    w_q, w_k, w_v = (torch.randn(D, D) / math.sqrt(D) for _ in range(3))

    # (a) no cache: recompute the full prefix at every step
    no_cache_last = []
    for t in range(1, seq_len + 1):
        prefix = x[:t]
        out = attention(prefix @ w_q, prefix @ w_k, prefix @ w_v, causal=True)
        no_cache_last.append(out[-1])

    # (b) with cache: one query, all cached keys/values
    key_cache, value_cache, cached_last = None, None, []
    for t in range(seq_len):
        token = x[t : t + 1]
        q, k, v = token @ w_q, token @ w_k, token @ w_v
        key_cache = k if key_cache is None else torch.cat([key_cache, k], dim=0)
        value_cache = v if value_cache is None else torch.cat([value_cache, v], dim=0)
        # A single query against the whole cache: no mask needed, there is no future.
        cached_last.append(attention(q, key_cache, value_cache, causal=False)[-1])

    difference = max((a - b).abs().max().item() for a, b in zip(no_cache_last, cached_last))
    print(f"  max difference over {seq_len} steps = {difference:.3e}  -> identical outputs")
    print("\n  Two details that matter in our implementation:")
    print("    - the cache grows along the SEQUENCE dimension: torch.cat(..., dim=-2)")
    print("      on tensors shaped [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]")
    print("    - during decoding q_len == 1, so there is no future to mask at all")

    print()
    print("=" * 74)
    print("3. PREFILL vs DECODE: TWO DIFFERENT WORKLOADS")
    print("=" * 74)
    print("  PREFILL (the first forward): the whole prompt at once, q_len = prompt length.")
    print("    Compute bound -- big matmuls, the GPU is happy.")
    print("  DECODE (every step after): q_len = 1.")
    print("    Memory bound -- we read every weight and the whole cache to produce one")
    print("    token, so the arithmetic intensity is terrible.")
    print("\n  That asymmetry is why the cache pays off and why the cache SIZE, not the")
    print("  FLOP count, is what limits your batch size in production.")

    print()
    print("=" * 74)
    print("4. THE COST IN OPERATIONS")
    print("=" * 74)
    print("  Generating n tokens after a prompt of length p, per layer:")
    print("    no cache : sum over t of O((p+t)^2 * d)   -- quadratic in the total length")
    print("    cache    : sum over t of O((p+t) * d)     -- linear")
    print("\n  Measured on this toy attention (d = 128):")
    for total in (128, 512, 1024):
        start = time.perf_counter()
        for t in range(1, total + 1):
            prefix = x[:1].expand(t, D)
            attention(prefix @ w_q, prefix @ w_k, prefix @ w_v, causal=True)
        no_cache_time = time.perf_counter() - start

        start = time.perf_counter()
        key_cache = value_cache = None
        for t in range(total):
            token = x[:1]
            k, v = token @ w_k, token @ w_v
            key_cache = k if key_cache is None else torch.cat([key_cache, k], dim=0)
            value_cache = v if value_cache is None else torch.cat([value_cache, v], dim=0)
            attention(token @ w_q, key_cache, value_cache, causal=False)
        cache_time = time.perf_counter() - start
        print(f"    {total:>5} tokens: no cache {no_cache_time * 1e3:8.1f} ms | "
              f"cache {cache_time * 1e3:7.1f} ms | {no_cache_time / cache_time:5.1f}x faster")

    print()
    print("=" * 74)
    print("5. THE COST IN MEMORY -- AND WHY GEMMA USES 1 KV HEAD")
    print("=" * 74)
    print("  cache bytes = 2 (K and V) x layers x kv_heads x head_dim x seq_len x dtype")
    layers, head_dim, dtype_bytes = 18, 256, 2  # paligemma-3b, bfloat16
    print(f"\n  paligemma-3b: {layers} layers, head_dim {head_dim}, bfloat16, seq_len 1024")
    for kv_heads, label in ((8, "if it used 8 KV heads (plain MHA)"), (1, "actual: 1 KV head (MQA)")):
        size = 2 * layers * kv_heads * head_dim * 1024 * dtype_bytes
        print(f"    {label:<34} {size / 1e6:8.2f} MB per sequence")
    print("\n  Multiply by your batch size and you see why Episode 18's grouped-query")
    print("  attention exists. For a 70B model with 80 layers and 8 KV heads at 4k")
    print("  tokens you are looking at gigabytes -- per user.")

    print()
    print("=" * 74)
    print("6. WHAT WE ARE ABOUT TO WRITE")
    print("=" * 74)
    print("  class KVCache:")
    print("      key_cache:   List[Tensor]   one entry per layer")
    print("      value_cache: List[Tensor]")
    print("      num_items()  -> cached sequence length (shape[-2] of layer 0)")
    print("      update(k, v, layer_idx) -> append along dim=-2, return the full K, V")
    print("\n  The KVCache instance lives in inference.py, outside the model, and is")
    print("  passed into every forward. num_items() == 0 is how the model knows it is")
    print("  in prefill rather than decode (Episode 15).")


if __name__ == "__main__":
    main()
