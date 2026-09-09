"""Episode 02 -- the CLIP contrastive loss, by hand.

We never train CLIP in this playlist, but you cannot understand *why* the vision
encoder produces the embeddings it produces without seeing the loss that shaped
them. Everything here runs on toy 4-dimensional embeddings so you can read the
numbers.

Run:  python clip_contrastive_loss.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


def clip_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07):
    """The symmetric InfoNCE loss of CLIP.

    image_embeds, text_embeds: [Batch_Size, Embed_Dim], one text per image, and the
    i-th text is the caption of the i-th image. Every off-diagonal pair is treated
    as a negative -- that is the whole trick, the batch *is* the label.
    """
    # L2 normalize, so a dot product is a cosine similarity in [-1, 1].
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # [Batch_Size, Batch_Size]: logits[i, j] = how much image i matches text j.
    logits = image_embeds @ text_embeds.t() / temperature

    # The correct answer for row i is column i, and for column j it is row j:
    # the targets are simply the indices on the diagonal.
    labels = torch.arange(logits.shape[0])

    # Cross entropy over the rows (image -> text retrieval) ...
    loss_i = F.cross_entropy(logits, labels)
    # ... and over the columns (text -> image retrieval). CLIP averages the two.
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2, logits


def show(matrix: torch.Tensor, title: str) -> None:
    print(f"\n{title}")
    print("        " + "".join(f"text{j}   " for j in range(matrix.shape[1])))
    for i, row in enumerate(matrix):
        cells = "".join(f"{value:+7.2f} " for value in row)
        print(f"image{i} {cells}")


def main() -> None:
    batch_size, embed_dim = 4, 8

    print("=" * 74)
    print("1. RANDOM ENCODERS: nothing is aligned yet")
    print("=" * 74)
    image_embeds = torch.randn(batch_size, embed_dim)
    text_embeds = torch.randn(batch_size, embed_dim)
    loss, logits = clip_loss(image_embeds, text_embeds)
    show(logits, "logits = image . text / temperature")
    print(f"\nloss = {loss:.4f}   (chance level = ln({batch_size}) = {torch.log(torch.tensor(float(batch_size))):.4f})")
    print("The diagonal is not special, so the loss sits at chance level.")

    print()
    print("=" * 74)
    print("2. A TRAINED PAIR OF ENCODERS: matching pairs point the same way")
    print("=" * 74)
    # Simulate what training achieves: the caption embedding is close to its image.
    text_embeds = image_embeds + 0.15 * torch.randn(batch_size, embed_dim)
    loss, logits = clip_loss(image_embeds, text_embeds)
    show(logits, "logits with aligned pairs")
    print(f"\nloss = {loss:.4f}")
    print("The diagonal now dominates each row AND each column, so the loss collapses.")

    print()
    print("=" * 74)
    print("3. WHY THE TEMPERATURE MATTERS")
    print("=" * 74)
    print("Temperature scales the logits before the softmax: low temperature")
    print("sharpens the distribution and punishes hard negatives much harder.\n")
    for temperature in (1.0, 0.5, 0.07, 0.01):
        loss, _ = clip_loss(image_embeds, text_embeds, temperature)
        print(f"  temperature = {temperature:<5} -> loss = {loss:.4f}")
    print("\nCLIP learns log(1/temperature) as a parameter instead of fixing it.")

    print()
    print("=" * 74)
    print("4. THE PROBLEM THAT LEADS TO SigLIP (Episode 04)")
    print("=" * 74)
    print("The softmax in row i needs *every* similarity in that row, and the softmax")
    print("in column j needs every similarity in that column. So the whole")
    print("[Batch_Size, Batch_Size] matrix must exist at once:")
    for n in (1_024, 8_192, 32_768):
        entries = n * n
        print(f"  batch = {n:>6} -> {entries:>12,} similarities, {entries * 4 / 1e9:6.2f} GB in fp32")
    print("With CLIP-style training you want the biggest batch you can afford, which")
    print("is exactly where this quadratic all-to-all matrix becomes the bottleneck.")


if __name__ == "__main__":
    main()
