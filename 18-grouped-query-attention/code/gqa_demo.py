"""Episode 18 -- multi-head vs multi-query vs grouped-query attention.

GQA is not an accuracy trick, it is a *memory bandwidth* trick. This script shows
what it costs, what it saves, and verifies our `repeat_kv` + `GemmaAttention` code.

Run:  python gqa_demo.py
"""

import math

import torch

from modeling_gemma import GemmaAttention, GemmaConfig, KVCache, repeat_kv

torch.manual_seed(0)


def main() -> None:
    print("=" * 74)
    print("1. THE THREE VARIANTS")
    print("=" * 74)
    print("  MHA (multi-head)   : num_kv_heads == num_q_heads. Every query head has its")
    print("                       own K and V. Maximum expressivity, maximum cache.")
    print("  MQA (multi-query)  : num_kv_heads == 1. All query heads share one K/V pair.")
    print("                       Smallest cache, slight quality loss.")
    print("  GQA (grouped-query): num_kv_heads == g, with 1 < g < num_q_heads. Each")
    print("                       group of query heads shares a K/V pair. The knob.")
    print("\n  Only the K and V projections shrink -- the number of QUERY heads, and")
    print("  therefore the number of attention patterns, never changes.")

    print()
    print("=" * 74)
    print("2. WHY: DECODING IS BOUND BY MEMORY, NOT BY MATH")
    print("=" * 74)
    print("  To generate ONE token a decoder must read every weight and the entire")
    print("  KV-cache from HBM, then do a trivial amount of arithmetic on it. Modern")
    print("  accelerators can do ~100-1000 FLOPs in the time it takes to fetch one")
    print("  byte, so the fetch is the bottleneck. Shrinking the cache directly buys")
    print("  you tokens per second -- and lets you fit more users in memory.")

    print()
    print("=" * 74)
    print("3. THE NUMBERS FOR paligemma-3b (18 layers, head_dim 256, bfloat16)")
    print("=" * 74)
    layers, head_dim, seq_len, dtype_bytes, q_heads = 18, 256, 2048, 2, 8
    print(f"  {'variant':<22}{'kv_heads':>9}{'cache @ 2048 tok':>20}{'k/v params/layer':>19}")
    for kv_heads, name in ((8, "MHA"), (4, "GQA (g=4)"), (2, "GQA (g=2)"), (1, "MQA -- Gemma's choice")):
        cache = 2 * layers * kv_heads * head_dim * seq_len * dtype_bytes
        kv_parameters = 2 * 2048 * kv_heads * head_dim
        print(f"  {name:<22}{kv_heads:>9}{cache / 1e6:>17.1f} MB{kv_parameters:>19,}")
    print(f"\n  {q_heads} query heads either way. MQA cuts the cache by {q_heads}x for free.")

    print()
    print("=" * 74)
    print("4. repeat_kv: HOW THE SHARING IS IMPLEMENTED")
    print("=" * 74)
    batch_size, kv_heads, seq, dim = 1, 2, 3, 4
    kv = torch.arange(batch_size * kv_heads * seq * dim, dtype=torch.float32)
    kv = kv.view(batch_size, kv_heads, seq, dim)
    repeated = repeat_kv(kv, n_rep=4)
    print(f"  input  {list(kv.shape)}  [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]")
    print(f"  output {list(repeated.shape)}  after repeat_kv(..., n_rep=4)")
    print(f"\n  head 0 of the input is copied to output heads: "
          f"{[h for h in range(repeated.shape[1]) if torch.equal(repeated[0, h], kv[0, 0])]}")
    print(f"  head 1 of the input is copied to output heads: "
          f"{[h for h in range(repeated.shape[1]) if torch.equal(repeated[0, h], kv[0, 1])]}")
    print("\n  So the layout is [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1]: CONSECUTIVE")
    print("  query heads form a group. Get this wrong (interleaving instead) and the")
    print("  model still runs, still produces plausible shapes, and quietly pairs each")
    print("  query with the wrong key. A very hard bug to spot -- worth a unit test.")
    print(f"\n  repeat_kv(kv, n_rep=1) returns the input untouched: "
          f"{repeat_kv(kv, 1) is kv}")
    print("  Note this materialises the expanded tensor, so it does NOT save the")
    print("  attention matmul any work -- the saving is in the cache and in the")
    print("  weights, not in the FLOPs.")

    print()
    print("=" * 74)
    print("5. OUR GemmaAttention, END TO END")
    print("=" * 74)
    config = GemmaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA with 4 query heads per group
        head_dim=8,
        pad_token_id=0,
    )
    attention = GemmaAttention(config, layer_idx=0).eval()
    print(f"  num_attention_heads {config.num_attention_heads}, num_key_value_heads"
          f" {config.num_key_value_heads} -> num_key_value_groups"
          f" {attention.num_key_value_groups}")
    print(f"  q_proj weight {list(attention.q_proj.weight.shape)}")
    print(f"  k_proj weight {list(attention.k_proj.weight.shape)}  <- {config.num_attention_heads // config.num_key_value_heads}x narrower")
    print(f"  v_proj weight {list(attention.v_proj.weight.shape)}")
    print(f"  o_proj weight {list(attention.o_proj.weight.shape)}")
    print(f"  bias: {config.attention_bias}  (Gemma uses no attention bias)")

    seq_len = 6
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0) + 1
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out, weights = attention(hidden_states, attention_mask, position_ids, kv_cache=None)
    print(f"\n  hidden_states {list(hidden_states.shape)} -> out {list(out.shape)}")
    print(f"  attn_weights {list(weights.shape)}  [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]")
    print(f"  rows sum to 1: {torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))}")

    print()
    print("=" * 74)
    print("6. THE CACHE IN ACTION (one layer, five steps)")
    print("=" * 74)
    kv_cache = KVCache()
    with torch.no_grad():
        for step in range(5):
            token = torch.randn(1, 1, config.hidden_size)
            position_ids = torch.tensor([[step + 1]])
            mask = torch.zeros(1, 1, 1, kv_cache.num_items() + 1)
            _, weights = attention(token, mask, position_ids, kv_cache=kv_cache)
            print(f"    step {step}: q_len 1, cache now holds {kv_cache.num_items()} positions, "
                  f"attn_weights {list(weights.shape)}")
    print(f"\n  key_cache[0] shape {list(kv_cache.key_cache[0].shape)}"
          f"  [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]")
    print("  Note the cache stores the NARROW tensors (2 kv heads). repeat_kv expands")
    print("  them to 8 on every forward, after the cache update -- storing the wide")
    print("  version would throw the saving away.")

    print()
    print("=" * 74)
    print("7. THE MASK IS ADDED, NOT MULTIPLIED")
    print("=" * 74)
    print("  `attn_weights = attn_weights + attention_mask` before the softmax, where")
    print("  the mask holds 0 (attend) or torch.finfo(dtype).min (block). We use the")
    print("  dtype minimum rather than -inf so that a fully masked row gives a uniform")
    print("  distribution instead of NaN.")
    print(f"    torch.finfo(float32).min = {torch.finfo(torch.float32).min:.3e}")
    print(f"    exp of that after shifting = {math.exp(-80):.2e} (effectively zero)")
    print("  And the softmax runs in float32 before being cast back -- Episode 03.")


if __name__ == "__main__":
    main()
