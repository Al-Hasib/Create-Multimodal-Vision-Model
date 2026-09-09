"""Episode 04 -- SigLIP: replace the softmax with a sigmoid.

CLIP asks "which of these N texts belongs to image i?" -- a classification over the
batch. SigLIP asks "do these two things belong together, yes or no?" for each pair
independently. That one change makes the loss decomposable, which is what lets you
shard it across devices.

Run:  python siglip_loss.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def clip_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Softmax over rows and columns -- needs the full matrix (Episode 02)."""
    logits = F.normalize(image_embeds, dim=-1) @ F.normalize(text_embeds, dim=-1).t() / temperature
    labels = torch.arange(logits.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    log_scale: float = 2.3026,  # t' in the paper, initialised so exp(t') = 10
    bias: float = -10.0,        # b in the paper, counteracts the many negatives
) -> torch.Tensor:
    """Pairwise binary cross entropy on the sigmoid of the scaled similarities.

    z_ij = +1 for a matching pair and -1 otherwise, and the loss of one pair is
    -log(sigmoid(z_ij * (t * x_i . y_j + b))). Note what is NOT here: no softmax,
    so no normalization across the batch.
    """
    batch_size = image_embeds.shape[0]
    logits = F.normalize(image_embeds, dim=-1) @ F.normalize(text_embeds, dim=-1).t()
    logits = logits * torch.exp(torch.tensor(log_scale)) + bias

    # +1 on the diagonal (positives), -1 everywhere else (negatives).
    signs = -torch.ones(batch_size, batch_size)
    signs.fill_diagonal_(1.0)

    # -log(sigmoid(signs * logits)), averaged the way the paper does it.
    return -F.logsigmoid(signs * logits).sum() / batch_size


def main() -> None:
    batch_size, embed_dim = 8, 16
    image_embeds = torch.randn(batch_size, embed_dim)
    aligned_texts = image_embeds + 0.15 * torch.randn(batch_size, embed_dim)
    random_texts = torch.randn(batch_size, embed_dim)

    print("=" * 74)
    print("1. BOTH LOSSES REWARD THE SAME THING")
    print("=" * 74)
    print(f"  random pairs  : CLIP = {clip_loss(image_embeds, random_texts):7.4f}   SigLIP = {siglip_loss(image_embeds, random_texts):7.4f}")
    print(f"  aligned pairs : CLIP = {clip_loss(image_embeds, aligned_texts):7.4f}   SigLIP = {siglip_loss(image_embeds, aligned_texts):7.4f}")
    print("  Different scales, same direction: matching pairs -> lower loss.")

    print()
    print("=" * 74)
    print("2. THE ASYMMETRY PROBLEM SigLIP HAS TO FIX")
    print("=" * 74)
    print(f"  In a batch of {batch_size} there are {batch_size} positive pairs and {batch_size * (batch_size - 1)} negative ones.")
    print("  Treating every pair as an independent yes/no question means the model can")
    print("  score well by answering 'no' to everything -- hence the learnable bias b,")
    print("  initialised to a large negative value so 'no' is the prior:\n")
    for bias in (0.0, -5.0, -10.0, -20.0):
        loss = siglip_loss(image_embeds, aligned_texts, bias=bias)
        print(f"    bias = {bias:>6} -> loss = {loss:7.4f}")

    print()
    print("=" * 74)
    print("3. THE REAL WIN: THE LOSS SPLITS INTO INDEPENDENT BLOCKS")
    print("=" * 74)
    print("  Because every pair contributes its own term, you can compute the loss")
    print("  chunk by chunk and add the pieces up. Let's prove it on this batch by")
    print("  splitting the pair matrix into 4x4 blocks:\n")

    log_scale, bias = 2.3026, -10.0
    scale = torch.exp(torch.tensor(log_scale))
    images = F.normalize(image_embeds, dim=-1)
    texts = F.normalize(aligned_texts, dim=-1)

    chunk = 4
    total = torch.tensor(0.0)
    for i in range(0, batch_size, chunk):
        for j in range(0, batch_size, chunk):
            block = images[i : i + chunk] @ texts[j : j + chunk].t() * scale + bias
            signs = -torch.ones(chunk, chunk)
            if i == j:  # the diagonal of the full matrix lives inside this block
                signs.fill_diagonal_(1.0)
            contribution = -F.logsigmoid(signs * block).sum() / batch_size
            total = total + contribution
            print(f"    block rows {i}-{i + chunk - 1}, cols {j}-{j + chunk - 1}: {contribution:8.4f}")

    full = siglip_loss(image_embeds, aligned_texts, log_scale, bias)
    print(f"\n    sum of blocks = {total:.6f}")
    print(f"    full loss     = {full:.6f}")
    print(f"    difference    = {(total - full).abs():.3e}  -> identical")

    print()
    print("  With CLIP you cannot do this: the softmax denominator of row i needs")
    print("  every column, so each device has to all-gather the full embedding matrix.")
    print("  SigLIP devices only ever exchange one chunk of embeddings at a time,")
    print("  which is how the paper trains with a batch size of 1 million.")

    print()
    print("=" * 74)
    print("4. WHY *THIS* ENCODER ENDS UP INSIDE PaliGemma")
    print("=" * 74)
    print("  We are not going to train anything -- we want the frozen vision tower.")
    print("  Contrastive pretraining is what makes its patch embeddings *language")
    print("  aligned*: they already live in a space where 'dog' and a picture of a dog")
    print("  are close, so a single linear projection (Episode 12) is enough to feed")
    print("  them to Gemma. An ImageNet classifier trained on 1000 labels would have")
    print("  thrown away everything the labels do not mention.")


if __name__ == "__main__":
    main()
