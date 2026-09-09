"""Episode 19 -- Rotary Positional Embedding, and proof that it is relative.

Every claim below is checked numerically against our own implementation.

Run:  python rope_demo.py
"""

import math

import torch

from modeling_gemma import GemmaRotaryEmbedding, apply_rotary_pos_emb, rotate_half

torch.manual_seed(0)


def rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, head_dim: int, base: float = 10000.0):
    """Apply our RoPE to q, k shaped [1, 1, Seq_Len, Head_Dim] at the given positions."""
    rotary = GemmaRotaryEmbedding(head_dim, base=base)
    cos, sin = rotary(q, positions)
    return apply_rotary_pos_emb(q, k, cos, sin)


def main() -> None:
    head_dim = 8

    print("=" * 74)
    print("1. ABSOLUTE vs RELATIVE, AND WHY WE WANT RELATIVE")
    print("=" * 74)
    print("  SigLIP adds a learned vector per slot (Episode 05). That is absolute:")
    print("  slot 5 has its own vector, unrelated to slot 6, and slot 5000 was never")
    print("  trained. What attention actually needs is the DISTANCE between a query")
    print("  and a key -- 'the previous word', 'three tokens back'.")
    print("\n  RoPE's idea: do not add anything. ROTATE q and k by an angle")
    print("  proportional to their position. The dot product of two rotated vectors")
    print("  then depends only on the difference of the angles.")

    print()
    print("=" * 74)
    print("2. THE FREQUENCIES")
    print("=" * 74)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    print(f"  theta_i = base^(-2i/dim) for i = 0..dim/2-1, base = 10000, dim = {head_dim}")
    print(f"  theta   = {[round(v, 5) for v in inv_freq.tolist()]}")
    print("\n  Each PAIR of dimensions is a 2D plane, rotated at its own speed:")
    for i, theta in enumerate(inv_freq.tolist()):
        wavelength = 2 * math.pi / theta
        print(f"    pair {i}: theta {theta:.5f}  ->  full turn every {wavelength:>10.1f} positions")
    print("  Fast pairs encode 'is this the token right before me', slow pairs encode")
    print("  'are we in the same paragraph'. A multi-resolution clock.")

    print()
    print("=" * 74)
    print("3. THE ROTATION, EXPLICITLY")
    print("=" * 74)
    print("  For a 2D pair (x1, x2) at position m, rotating by m*theta is:")
    print("      x1' = x1*cos(m*theta) - x2*sin(m*theta)")
    print("      x2' = x1*sin(m*theta) + x2*cos(m*theta)")
    print("  which is exactly what `x * cos + rotate_half(x) * sin` computes -- with")
    print("  one twist. Our rotate_half is:")
    x = torch.arange(1, head_dim + 1, dtype=torch.float32)
    print(f"    x              = {x.tolist()}")
    print(f"    rotate_half(x) = {rotate_half(x).tolist()}")
    print("\n  So the pairs are (0, 4), (1, 5), (2, 6), (3, 7) -- dimension i is paired")
    print("  with dimension i + dim/2, NOT with i+1. That is the Hugging Face layout,")
    print("  and it works because `cos`/`sin` are built as cat(freqs, freqs), so both")
    print("  halves of a pair get the same angle. The original paper interleaves the")
    print("  pairs instead; the two are related by a permutation of the head dimension")
    print("  (the checkpoint's weights are stored to match, so we must not 'fix' it).")

    print()
    print("=" * 74)
    print("4. PROOF: THE DOT PRODUCT ONLY SEES THE DISTANCE")
    print("=" * 74)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    print("  Same q, same k, several (query position, key position) pairs:\n")
    print(f"    {'q at':>6}{'k at':>6}{'distance':>10}{'q . k after RoPE':>20}")
    for query_position, key_position in ((5, 3), (6, 4), (100, 98), (5, 4), (5, 0)):
        q_rot, _ = rope(q, k, torch.tensor([[query_position]]), head_dim)
        _, k_rot = rope(q, k, torch.tensor([[key_position]]), head_dim)
        score = (q_rot * k_rot).sum().item()
        print(f"    {query_position:>6}{key_position:>6}{query_position - key_position:>10}{score:>20.6f}")
    print("\n  Rows 1-3 all have distance 2 and give the SAME score, at position 5 and")
    print("  at position 100. Nothing in the model had to learn that; it is a property")
    print("  of rotations. This is the whole point of RoPE.")

    print(f"\n  It also preserves the norm (a rotation cannot stretch a vector):")
    q_rot, _ = rope(q, k, torch.tensor([[42]]), head_dim)
    print(f"    ||q|| = {q.norm():.6f}   ||RoPE(q, 42)|| = {q_rot.norm():.6f}")

    print()
    print("=" * 74)
    print("5. LONG-TERM DECAY")
    print("=" * 74)
    print("  Careful with this one, it is easy to state wrongly. For two INDEPENDENT")
    print("  random vectors, rotating them changes nothing on average -- the dot")
    print("  product of isotropic noise is rotation invariant. Let's confirm that")
    print("  first, so the real effect is not mistaken for magic:\n")
    trials, real_head_dim = 2000, 256  # the actual head_dim of paligemma-3b
    q = torch.randn(1, 1, trials, real_head_dim)
    k = torch.randn(1, 1, trials, real_head_dim)
    reference_positions = torch.zeros(1, trials, dtype=torch.long)
    _, k_at_zero = rope(q, k, reference_positions, real_head_dim)
    for distance in (0, 16, 1024):
        q_rot, _ = rope(q, k, torch.full((1, trials), distance), real_head_dim)
        score = (q_rot * k_at_zero).sum(-1).abs().mean().item()
        print(f"    independent q, k -- distance {distance:>5}: mean |q . k| = {score:7.2f}   (flat)")

    print("\n  The decay appears for a query that MATCHES its key (q = k), which is the")
    print("  case attention actually cares about: a strong match nearby should score")
    print("  higher than the same match far away.\n")
    _, k_at_zero = rope(q, q, reference_positions, real_head_dim)
    for distance in (0, 1, 4, 16, 64, 256, 1024, 4096):
        q_rot, _ = rope(q, q, torch.full((1, trials), distance), real_head_dim)
        score = (q_rot * k_at_zero).sum(-1).mean().item()
        print(f"    matching q, k  -- distance {distance:>5}: mean q . k   = {score:7.2f}")
    print("\n  Maximal at distance 0, then down to near zero. The score is a sum of")
    print("  cos(distance * theta_i) weighted by the energy in each pair: at distance 0")
    print("  every cosine is 1 and they add up, and as the distance grows the fast")
    print("  pairs fall out of phase and cancel. The paper proves an upper bound with")
    print("  the same shape (Section 3.4.3).")
    print("\n  It is a soft prior, not a mask: attention can still learn to look far")
    print("  back, it just does not start out doing so.")

    print()
    print("=" * 74)
    print("6. WHERE IT IS APPLIED -- AND WHERE IT IS NOT")
    print("=" * 74)
    print("  In GemmaAttention.forward, RoPE is applied to Q and K only, AFTER the")
    print("  projections and BEFORE the cache update. Not to V.")
    print("    - not to V, because V is the payload, not part of the matching. Rotating")
    print("      it would corrupt the values we are averaging.")
    print("    - before the cache, because the cached K must already carry its position:")
    print("      it is written once and re-read at every later step.")
    print("    - inside every layer, because there is no single 'position embedding")
    print("      added at the input' to inherit.")
    print("\n  And it has NO parameters: `inv_freq` is a non-persistent buffer, computed")
    print("  from base and dim. It is not in the checkpoint and never trained.")
    rotary = GemmaRotaryEmbedding(head_dim)
    print(f"    parameters in GemmaRotaryEmbedding: {sum(p.numel() for p in rotary.parameters())}")
    print(f"    'inv_freq' in state_dict           : {'inv_freq' in rotary.state_dict()}")

    print()
    print("=" * 74)
    print("7. THE SHAPES, AND THE unsqueeze_dim=1 DETAIL")
    print("=" * 74)
    seq_len = 4
    q = torch.randn(1, 8, seq_len, head_dim)  # [Batch, Num_Heads_Q, Seq_Len, Head_Dim]
    k = torch.randn(1, 1, seq_len, head_dim)  # [Batch, Num_Heads_KV, Seq_Len, Head_Dim]
    position_ids = torch.arange(seq_len).unsqueeze(0) + 1
    cos, sin = GemmaRotaryEmbedding(head_dim)(q, position_ids)
    print(f"  cos, sin from the rotary module : {list(cos.shape)}  [Batch_Size, Seq_Len, Head_Dim]")
    print(f"  after unsqueeze(1)              : {list(cos.unsqueeze(1).shape)}  <- a head axis of size 1")
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"  q {list(q.shape)} -> {list(q_rot.shape)}")
    print(f"  k {list(k.shape)} -> {list(k_rot.shape)}")
    print("  One cos/sin table broadcasts over all heads: every head rotates by the")
    print("  same angle, which is why q (8 heads) and k (1 head) can share it.")

    print()
    print("=" * 74)
    print("8. base=10000 AND CONTEXT EXTENSION")
    print("=" * 74)
    print("  rope_theta (the base) sets the slowest wavelength, so it caps the range of")
    print(f"  distances the model can distinguish (shown for the real head_dim = {real_head_dim}):")
    for base in (10_000.0, 100_000.0, 1_000_000.0):
        slowest = 2 * math.pi * (base ** ((real_head_dim - 2) / real_head_dim))
        print(f"    base {base:>10,.0f} -> slowest pair turns once every {slowest:>12,.0f} positions")
    print("  Raising the base after training ('NTK scaling', 'rope scaling') is the")
    print("  standard way to extend a model's context window, and it works precisely")
    print("  because RoPE is a function of the position, not a learned table.")
    print("  Gemma keeps rope_theta = 10000 with max_position_embeddings = 8192.")

    print()
    print("=" * 74)
    print("9. THE TEXT MODEL IS NO LONGER POSITION-BLIND")
    print("=" * 74)
    from modeling_gemma import GemmaAttention, GemmaConfig

    config = GemmaConfig(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        pad_token_id=0,
    )
    attention = GemmaAttention(config, layer_idx=0).eval()
    hidden_states = torch.randn(1, 3, config.hidden_size)
    mask = torch.zeros(1, 1, 3, 3)
    with torch.no_grad():
        early, _ = attention(hidden_states, mask, torch.tensor([[1, 2, 3]]))
        shifted, _ = attention(hidden_states, mask, torch.tensor([[101, 102, 103]]))
        reversed_order, _ = attention(hidden_states, mask, torch.tensor([[3, 2, 1]]))
        spread_out, _ = attention(hidden_states, mask, torch.tensor([[1, 50, 300]]))
    print("  Same three token vectors, four different sets of position ids:\n")
    print(f"    1,2,3  vs  101,102,103 : max diff {(early - shifted).abs().max():.4f}   <- IDENTICAL")
    print(f"    1,2,3  vs  3,2,1       : max diff {(early - reversed_order).abs().max():.4f}")
    print(f"    1,2,3  vs  1,50,300    : max diff {(early - spread_out).abs().max():.4f}")
    print("\n  The first row is the payoff of section 4, not a bug: shifting every")
    print("  position by the same amount leaves every *distance* unchanged, so the")
    print("  output must be bit-identical. A learned absolute table would have")
    print("  produced three completely different results.")
    print("  The other rows change the distances, so the output changes.")
    print("\n  In Episode 18 all three differences would have been 0.0000, because the")
    print("  rotary stub returned cos = 1 and sin = 0. Position information enters the")
    print("  text model here and nowhere else.")
    print("\n  The vision tower gets its positions from a learned table, the text model")
    print("  gets them from these rotations, and the model is now complete:")
    print("  Episode 20 loads the real weights and generates text.")


if __name__ == "__main__":
    main()
