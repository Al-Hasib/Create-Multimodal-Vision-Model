# Episode 02 — Contrastive Learning & CLIP

> **Video chapter:** `00:05:52 – 00:16:50` (Contrastive Learning and CLIP)
> **Target runtime:** 15–18 min · **Code:** [`code/clip_contrastive_loss.py`](code/clip_contrastive_loss.py)

## Why start with a loss function we never use?

We are not going to train anything. But the vision encoder we load in Episode 10 is the
*product* of a contrastive loss, and that loss is the reason its output can be fed to a
language model at all. Understand the loss and the architecture stops looking arbitrary.

## The problem

We want an image encoder whose embeddings mean something *in language terms*. Two obvious
options fail:

- **Train a classifier on ImageNet.** Its features are shaped by 1000 labels. Everything
  the labels do not mention — the colour of the sky, the mood, the text on the sign — gets
  discarded, because discarding it does not hurt the loss.
- **Hand-label captions.** Nobody is labelling 400 million images.

The web already has the labels: images with alt-text, captions, surrounding paragraphs.
Noisy, but free and enormous. All we need is a loss that can learn from *pairs*.

## The idea: learn a shared space

Two encoders, one for images and one for text, both projecting into the same
`Embed_Dim`-dimensional space. Train them so that **an image and its own caption land
close together, and every other pairing lands far apart.**

```
   image_i ──► image encoder ──► I_i ┐
                                     ├──►  I_i · T_j   ("how well do these match?")
   text_j  ──► text encoder  ──► T_j ┘
```

The labels come from the batch itself: in a batch of N pairs, `(i, i)` is a positive and
all N²−N off-diagonal pairs are negatives. No annotation required — this is why
contrastive learning scales.

## The CLIP loss, precisely

1. **L2-normalize** both embeddings, so a dot product is a cosine similarity in [−1, 1].
2. **Similarity matrix** `logits = I @ T.T / temperature`, shape `[N, N]`.
3. **Symmetric cross-entropy.** For row `i` the correct class is column `i`, so
   `cross_entropy(logits, arange(N))` is image→text retrieval. Transpose and do it again
   for text→image. Average the two.

```python
logits = normalize(image_embeds) @ normalize(text_embeds).t() / temperature
labels = torch.arange(N)
loss = (cross_entropy(logits, labels) + cross_entropy(logits.t(), labels)) / 2
```

Two things deserve attention.

**Why both directions?** One-directional loss is satisfied by a degenerate solution:
if every text embedding is identical, image→text is hard but text→image is trivially
"whatever image, same text". Enforcing it both ways rules that out.

**What the temperature does.** It scales the logits before the softmax. Cosine similarity
lives in [−1, 1], so raw logits are far too flat for a sharp softmax; dividing by 0.07
stretches them to [−14, 14]. Low temperature = confident distribution = hard negatives get
punished much harder. CLIP does not fix it, it *learns* it (as `log(1/t)`, so it stays
positive), and clips it to prevent the model from cranking it up indefinitely.

Run the script to see the loss drop from `ln(N)` (chance) to near zero as the pairs align,
and to see what the temperature does to the same batch.

## What this bought us

Zero-shot classification, for free. To classify an image into `{cat, dog, car}`, embed the
three strings `"a photo of a cat"`, …, embed the image, and take the nearest. No training
head, no fine-tuning — the class names are just text. That is the CLIP paper's headline
result, and the property we exploit: **the image embeddings already live in a
language-shaped space.**

## The problem that leads to SigLIP

The softmax in row `i` needs every entry of row `i`; the softmax in column `j` needs every
entry of column `j`. So the entire `[N, N]` matrix has to exist at once, and every device
in a distributed run needs *all* embeddings — an all-gather per step:

| batch size | similarities | fp32 memory |
|---|---|---|
| 1,024 | 1.0 M | 4 MB |
| 8,192 | 67 M | 268 MB |
| 32,768 | 1.07 B | 4.3 GB |

And contrastive learning *wants* huge batches, because the batch is where the negatives
come from. That quadratic all-to-all coupling is the bottleneck Episode 04 removes.

## Run it

```bash
cd 02-contrastive-learning-and-clip/code
python clip_contrastive_loss.py
```

## Key takeaways

- The batch supplies the labels: diagonal = positive, off-diagonal = negative.
- Loss is symmetric cross-entropy over rows *and* columns; one direction alone has a
  degenerate solution.
- Temperature is a learned parameter, not a constant.
- Contrastive features are language-aligned, which is exactly what a VLM needs.
- The softmax couples the whole batch — remember this for Episode 04.

## Gotchas

- **Forgetting to normalize** turns cosine similarity into an unbounded dot product, and
  the model can win by inflating vector norms instead of aligning directions.
- **`cross_entropy` expects logits, not probabilities.** Feeding it softmax output applies
  the softmax twice, and the gradients get very small. Related to Episode 03.
- **Small batches make the task too easy** — with N=2 the model gets 50% by guessing.

## Exercises

1. Make the loss one-directional (rows only) and construct by hand a pair of embedding
   matrices that achieve a near-zero loss while being useless for text→image retrieval.
2. Sweep the temperature from 1.0 to 0.01 on *misaligned* embeddings. Why does the loss
   get *worse* as the temperature drops? What does that say about hard negatives?
3. Cosine similarity is bounded by 1, so the best achievable logit is `1/temperature`.
   Compute the lowest possible loss for N=1024 at `t = 0.07`. Is zero reachable?

## Further reading

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*
  (CLIP), <https://arxiv.org/abs/2103.00020>
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding*
  (the InfoNCE loss), <https://arxiv.org/abs/1807.03748>

**Previous:** [Episode 01](../01-introduction-and-setup/) ·
**Next:** [Episode 03 — Numerical stability of the softmax](../03-numerical-stability-of-softmax/)
