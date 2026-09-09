"""Episode 03 -- why every softmax in this repo subtracts the maximum.

Run:  python softmax_stability.py
"""

import torch


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """The formula straight from the textbook: exp(x_i) / sum_j exp(x_j)."""
    exponentials = torch.exp(x)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    """The same function, shifted by the row maximum.

    softmax(x) == softmax(x - c) for any constant c, because exp(x_i - c) / sum_j
    exp(x_j - c) = (exp(x_i) / exp(c)) / (sum_j exp(x_j) / exp(c)). Choosing
    c = max(x) makes the largest exponent exactly exp(0) = 1, so nothing overflows,
    and at least one term in the denominator is 1, so nothing underflows to 0/0.
    """
    maximum = x.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(x - maximum)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


def main() -> None:
    print("=" * 74)
    print("1. WHERE exp() BREAKS")
    print("=" * 74)
    for dtype in (torch.float16, torch.float32):
        info = torch.finfo(dtype)
        limit = torch.log(torch.tensor(info.max, dtype=torch.float32))
        print(f"  {str(dtype):<15} largest value {info.max:<12.4g} -> exp(x) overflows for x > {limit:.1f}")
    print("\nAttention logits are dot products of long vectors, so values in the")
    print("hundreds are entirely normal -- especially in float16.")

    print()
    print("=" * 74)
    print("2. THE NAIVE VERSION BLOWS UP")
    print("=" * 74)
    logits = torch.tensor([[900.0, 901.0, 902.0]])
    print(f"  logits         : {logits.tolist()[0]}")
    print(f"  naive_softmax  : {naive_softmax(logits).tolist()[0]}   <- inf / inf = nan")
    print(f"  stable_softmax : {stable_softmax(logits).tolist()[0]}")
    print(f"  torch.softmax  : {torch.softmax(logits, dim=-1).tolist()[0]}")

    print()
    print("  ...and it also breaks in the other direction (everything very negative):")
    logits = torch.tensor([[-900.0, -901.0, -902.0]])
    print(f"  logits         : {logits.tolist()[0]}")
    print(f"  naive_softmax  : {naive_softmax(logits).tolist()[0]}   <- 0 / 0 = nan")
    print(f"  stable_softmax : {stable_softmax(logits).tolist()[0]}")

    print()
    print("=" * 74)
    print("3. THE TWO AGREE WHENEVER THE NAIVE ONE SURVIVES")
    print("=" * 74)
    torch.manual_seed(0)
    logits = torch.randn(1000, 64) * 3
    difference = (naive_softmax(logits) - stable_softmax(logits)).abs().max()
    print(f"  max |naive - stable| over 1000 random rows = {difference:.3e}")
    print("  Subtracting the max is free accuracy: same function, no overflow.")

    print()
    print("=" * 74)
    print("4. THE SAME TRICK FOR log(softmax): LOG-SUM-EXP")
    print("=" * 74)
    logits = torch.tensor([[900.0, 901.0, 902.0]])
    print(f"  log(naive_softmax(x))         : {torch.log(naive_softmax(logits)).tolist()[0]}")
    maximum = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = maximum + torch.log(torch.exp(logits - maximum).sum(dim=-1, keepdim=True))
    print(f"  x - logsumexp(x)              : {(logits - log_sum_exp).tolist()[0]}")
    print(f"  torch.log_softmax(x)          : {torch.log_softmax(logits, dim=-1).tolist()[0]}")
    print("\n  This is why losses are computed from logits, never from probabilities:")
    print("  cross_entropy(logits, y) does the shift internally, log(softmax(x)) cannot.")

    print()
    print("=" * 74)
    print("5. WHAT THIS COSTS ATTENTION -- AND WHAT FLASH ATTENTION DOES ABOUT IT")
    print("=" * 74)
    print("  The shift needs max(row), and the denominator needs sum(row), so a plain")
    print("  softmax reads the whole row of the [Seq_Len, Seq_Len] score matrix twice.")
    print("  That matrix, not the model weights, is what dominates memory:")
    for seq_len in (256, 1_024, 8_192):
        entries = seq_len * seq_len
        print(f"    seq_len = {seq_len:>5} -> {entries:>12,} scores per head, {entries * 4 / 1e6:8.2f} MB in fp32")
    print("  Flash Attention keeps the max and the sum as running statistics and")
    print("  rescales as it walks the row block by block, so the full matrix never")
    print("  reaches memory. Same numbers, same stability trick, one pass.")


if __name__ == "__main__":
    main()
