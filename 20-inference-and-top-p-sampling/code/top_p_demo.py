"""Episode 20 -- temperature and top-p (nucleus) sampling.

`_sample_top_p` in inference.py is six lines of tensor gymnastics. This script takes
them apart on a 10-token vocabulary.

Run:  python top_p_demo.py
"""

import torch

torch.manual_seed(0)

VOCAB = ["cat", "dog", "car", "the", "and", "sky", "run", "red", "sun", "sea"]


def sample_top_p(probs: torch.Tensor, p: float):
    """The exact function from inference.py, instrumented so we can watch it."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Subtracting probs_sort shifts the cumulative sum one place right, so the
    # token that crosses p is KEPT rather than cut.
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token), probs_sort, probs_idx, mask


def main() -> None:
    logits = torch.tensor([[6.0, 5.5, 4.0, 3.5, 3.0, 1.0, 0.5, 0.0, -1.0, -2.0]])

    print("=" * 74)
    print("1. TEMPERATURE FIRST")
    print("=" * 74)
    print("  inference.py does `softmax(logits / temperature)` before top-p.")
    print(f"  {'token':<8}" + "".join(f"{f'T={t}':>10}" for t in (0.1, 0.5, 1.0, 2.0)))
    columns = [torch.softmax(logits / t, dim=-1)[0] for t in (0.1, 0.5, 1.0, 2.0)]
    for i, word in enumerate(VOCAB):
        print(f"  {word:<8}" + "".join(f"{column[i]:>10.4f}" for column in columns))
    print("\n  T < 1 sharpens (more deterministic), T > 1 flattens (more random).")
    print("  T -> 0 approaches argmax, which is what do_sample=False does directly.")

    print()
    print("=" * 74)
    print("2. WHY NOT JUST SAMPLE FROM THE FULL DISTRIBUTION?")
    print("=" * 74)
    probs = torch.softmax(logits, dim=-1)[0]
    tail = probs[5:].sum()
    print(f"  Here the 5 worst tokens hold only {tail:.2%} of the mass -- negligible.")
    print("  But a 257,152-token vocabulary has a LOT of tail. Sampled often enough,")
    print("  garbage does come out, and one bad token derails everything after it.")
    print("  Top-p keeps the smallest set of tokens whose mass reaches p, and")
    print("  redistributes. Unlike top-k it adapts: a confident step keeps 1-2 tokens,")
    print("  an uncertain step keeps many.")

    print()
    print("=" * 74)
    print("3. THE FUNCTION, STEP BY STEP (p = 0.9, T = 1)")
    print("=" * 74)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    shifted = cumulative - sorted_probs
    print(f"  {'rank':<6}{'token':<8}{'prob':>9}{'cumsum':>9}{'cumsum-prob':>13}{'  > 0.9 ?'}")
    for rank in range(len(VOCAB)):
        token = VOCAB[sorted_idx[0, rank]]
        cut = "  DROPPED" if shifted[0, rank] > 0.9 else "  kept"
        print(f"  {rank:<6}{token:<8}{sorted_probs[0, rank]:>9.4f}{cumulative[0, rank]:>9.4f}"
              f"{shifted[0, rank]:>13.4f}{cut}")
    print("\n  The `- probs_sort` is the subtle bit: comparing the cumulative sum")
    print("  BEFORE adding a token means the token that pushes past p is still kept,")
    print("  so the kept mass is always >= p and the set is never empty.")

    print()
    print("=" * 74)
    print("4. WHAT p DOES TO THE CANDIDATE SET")
    print("=" * 74)
    for p in (0.3, 0.6, 0.9, 0.99, 1.0):
        _, renormalized, indices, _ = sample_top_p(torch.softmax(logits, dim=-1).clone(), p)
        kept = [VOCAB[indices[0, r]] for r in range(len(VOCAB)) if renormalized[0, r] > 0]
        print(f"  p = {p:<5} keeps {len(kept)} token(s): {kept}")
    print("\n  Note the probabilities are re-normalized to sum to 1 after the cut --")
    print("  torch.multinomial needs a valid distribution.")

    print()
    print("=" * 74)
    print("5. THE gather AT THE END")
    print("=" * 74)
    print("  multinomial samples an index into the SORTED array, not a token id.")
    print("  `torch.gather(probs_idx, -1, next_token)` maps it back to the vocabulary.")
    print("  Forget this line and your model will generate fluent nonsense from the")
    print("  wrong token ids -- another silent bug worth remembering.")

    print()
    print("=" * 74)
    print("6. 10,000 SAMPLES, TO SEE IT ACTUALLY WORK")
    print("=" * 74)
    counts = torch.zeros(len(VOCAB))
    for _ in range(10_000):
        token, _, _, _ = sample_top_p(torch.softmax(logits, dim=-1).clone(), 0.9)
        counts[token.item()] += 1
    print(f"  {'token':<8}{'sampled':>10}{'full softmax':>14}")
    full = torch.softmax(logits, dim=-1)[0]
    for i, word in enumerate(VOCAB):
        print(f"  {word:<8}{counts[i] / 10_000:>10.4f}{full[i]:>14.4f}")
    print("\n  The kept tokens are slightly over-represented compared with the raw")
    print("  softmax (they absorbed the dropped mass), and the tail is exactly 0.")


if __name__ == "__main__":
    main()
