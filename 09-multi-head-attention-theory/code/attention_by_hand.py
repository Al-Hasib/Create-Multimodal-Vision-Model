"""Episode 09 -- multi-head attention with numbers small enough to read.

No nn.Module here: just tensors, so you can follow every shape change. This is the
same computation we are about to write inside `SiglipAttention` (Episode 10) and
`GemmaAttention` (Episode 18).

Run:  python attention_by_hand.py
"""

import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)


def main() -> None:
    seq_len, embed_dim, num_heads = 4, 8, 2
    head_dim = embed_dim // num_heads

    print("=" * 74)
    print("1. ONE HEAD, STEP BY STEP")
    print("=" * 74)
    x = torch.randn(seq_len, embed_dim)
    w_q, w_k, w_v = (torch.randn(embed_dim, embed_dim) / math.sqrt(embed_dim) for _ in range(3))

    q, k, v = x @ w_q, x @ w_k, x @ w_v
    print(f"  x {list(x.shape)} -> Q {list(q.shape)}  K {list(k.shape)}  V {list(v.shape)}")
    print("  Q = 'what am I looking for', K = 'what do I offer', V = 'what I pass on'")

    scores = q @ k.t()
    print(f"\n  scores = Q @ K^T  {list(scores.shape)}   (row i = how much token i wants token j)")
    print(scores)

    print(f"\n  Why divide by sqrt(head_dim) = sqrt({head_dim}) = {math.sqrt(head_dim):.3f}?")
    print("  A dot product of two d-dimensional unit-variance vectors has variance d,")
    print("  so scores grow with the head dimension and push the softmax into")
    print("  saturation, where the gradient is ~0. Let's measure it:")
    for d in (8, 64, 512):
        a, b = torch.randn(20_000, d), torch.randn(20_000, d)
        raw = (a * b).sum(-1)
        print(f"    head_dim {d:>4}: std(q.k) = {raw.std():7.2f}   after scaling = {(raw / math.sqrt(d)).std():.2f}")

    weights = torch.softmax(scores / math.sqrt(head_dim), dim=-1)
    print(f"\n  weights = softmax(scores / sqrt(d_k), dim=-1)   rows sum to {weights.sum(-1).tolist()}")
    print(weights)
    print("  dim=-1 is the whole point: each *query* gets a probability distribution")
    print("  over the keys. Softmax over dim=-2 would be a different (wrong) model.")

    out = weights @ v
    print(f"\n  output = weights @ V  {list(out.shape)}: a weighted average of the value")
    print("  vectors. Token i's output is 'the mixture of the tokens I cared about'.")

    print()
    print("=" * 74)
    print("2. WHY MORE THAN ONE HEAD")
    print("=" * 74)
    print("  A single softmax row is one distribution: it can attend to one thing at a")
    print("  time. Language needs several relations at once (syntax, coreference,")
    print("  position). So we split the embedding into `num_heads` slices and run")
    print("  independent attentions on each, then concatenate.")
    print(f"\n  embed_dim {embed_dim} = num_heads {num_heads} x head_dim {head_dim}")
    print("  Cost is unchanged: h heads of size d/h do the same FLOPs as 1 head of size d.")

    print()
    print("=" * 74)
    print("3. THE FOUR-LINE SHAPE DANCE WE WILL WRITE IN CODE")
    print("=" * 74)
    batch_size = 2
    x = torch.randn(batch_size, seq_len, embed_dim)
    q = x @ w_q
    print(f"  q_proj(x)                                              {list(q.shape)}")
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    print(f"  .view(B, Seq, Num_Heads, Head_Dim)                     {list(q.shape)}")
    q = q.transpose(1, 2)
    print(f"  .transpose(1, 2)                                       {list(q.shape)}  <- heads become a batch dim")
    print("\n  The transpose is what makes the heads independent: matmul only touches")
    print("  the last two dimensions, so every (batch, head) pair is its own attention.")
    k = (x @ w_k).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = (x @ w_v).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    attn = torch.softmax(q @ k.transpose(2, 3) / math.sqrt(head_dim), dim=-1)
    print(f"  Q @ K^T                                                {list(attn.shape)}  [B, H, Seq_Q, Seq_KV]")
    out = attn @ v
    print(f"  @ V                                                    {list(out.shape)}")
    out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, embed_dim)
    print(f"  .transpose(1, 2).reshape(B, Seq, Embed_Dim)            {list(out.shape)}  <- heads concatenated")
    print("\n  `.contiguous()` is required because transpose only changes the strides,")
    print("  and `.reshape`/`.view` needs a contiguous buffer to merge dimensions.")
    print("  Finally out_proj (W_o) mixes the heads together -- without it the heads")
    print("  would never talk to each other.")

    print(f"\n  Cross-check against PyTorch's fused kernel:")
    reference = F.scaled_dot_product_attention(q, k, v)
    ours = attn @ v
    print(f"    max |ours - F.scaled_dot_product_attention| = {(ours - reference).abs().max():.3e}")

    print()
    print("=" * 74)
    print("4. MASKS: THE ONLY DIFFERENCE BETWEEN OUR TWO ATTENTIONS")
    print("=" * 74)
    scores = torch.randn(seq_len, seq_len)
    causal = torch.full((seq_len, seq_len), float("-inf")).triu(diagonal=1)
    print("  A mask is added to the scores BEFORE the softmax, using -inf so that")
    print("  exp(-inf) = 0 removes the token exactly:")
    print(causal)
    print("\n  masked softmax:")
    print(torch.softmax(scores + causal, dim=-1))
    print("\n  SigLIP (image)  : no mask -- every patch sees every patch.")
    print("  Gemma  (text)   : causal mask -- token i may not see the future.")
    print("  PaliGemma prefix: the image tokens plus the prompt are NOT causally")
    print("                    masked (Episode 15) -- a detail specific to this model.")


if __name__ == "__main__":
    main()
