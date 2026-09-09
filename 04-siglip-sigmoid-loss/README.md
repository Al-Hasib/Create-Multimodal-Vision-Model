# Episode 04 — SigLIP: the Sigmoid Loss (and why a contrastive encoder)

> **Video chapters:** `00:23:00 – 00:29:13` (SigLip · Why a Contrastive Vision Encoder?)
> **Target runtime:** 12–15 min · **Code:** [`code/siglip_loss.py`](code/siglip_loss.py)

## The one-sentence version

CLIP asks *"which of these N captions belongs to image i?"* — a classification over the
batch. SigLIP asks *"do these two belong together, yes or no?"* — independently, for every
pair. Swap the softmax for a sigmoid and the batch stops being a bottleneck.

## The loss

For each of the N² pairs in a batch:

```
loss_ij = -log σ( z_ij · (t · x_i·y_j + b) )       z_ij = +1 if i == j else -1
```

- `x_i·y_j` — cosine similarity of image `i` and text `j` (both L2-normalized)
- `t = exp(t')` — a learnable scale, `t'` initialised so `t ≈ 10`
- `b` — a learnable bias, initialised to about −10
- `σ` — the logistic sigmoid

```python
logits = normalize(image_embeds) @ normalize(text_embeds).t() * t + b
signs = -torch.ones(N, N); signs.fill_diagonal_(1.0)
loss = -F.logsigmoid(signs * logits).sum() / N
```

Note what is *absent*: no softmax, therefore **no normalization across the batch.** Each
term is a self-contained binary cross-entropy on one pair.

## Why the bias `b` has to be there

A batch of N has N positives and N²−N negatives. At N = 1024 that is 1023 negatives per
positive. Treating every pair as an independent yes/no question means "always say no"
scores extremely well, and early training is dominated by pushing everything apart. The
large negative `b` bakes that prior in from the start — "assume no unless the similarity is
convincing" — so the gradient signal is not swamped. It is learnable, so the model can
adjust as it improves. The script sweeps `b` and you can watch the loss respond.

## The real win: the loss decomposes

This is the part worth internalising, and the script *proves* it numerically by splitting
a batch into blocks and summing the pieces — exactly matching the full loss.

**CLIP:** the softmax denominator of row `i` contains every column. To compute one row you
need every text embedding in the batch. Distributed across D devices, each device must
all-gather all N embeddings, hold an `[N, N]` matrix, and do it again for the columns.

**SigLIP:** each pair contributes its own additive term, so a device holding a chunk of
images and a chunk of texts can compute that block alone. The devices pass chunks around
in a ring, each computing its blocks, and the loss is the sum. Memory per device is
`[chunk, chunk]`, not `[N, N]`, and communication is one chunk at a time.

That is how the SigLIP paper trains with a batch size of **one million** — and, more
usefully, why SigLIP *beats* CLIP at small and medium batch sizes too: with a sigmoid loss
you no longer need a giant batch for the loss to be well-behaved.

## Why PaliGemma uses a *contrastive* vision encoder

We are not training. Why does it matter which loss shaped the encoder we load?

Because contrastive pretraining is what makes the patch embeddings **language-aligned.**
They already live in a space organised by linguistic meaning — a picture of a dog sits near
the text "dog". So the bridge into the language model can be a *single linear layer*
(Episode 13), and the language model can make sense of what arrives.

Compare the alternatives:

| Encoder | Trained on | Problem |
|---|---|---|
| ImageNet classifier | 1000 labels | features keep only what the labels need |
| MAE / autoencoder | pixel reconstruction | features encode texture and layout, not meaning |
| **Contrastive (CLIP/SigLIP)** | image–text pairs | features are already language-shaped |

There is a second, quieter reason: contrastive encoders keep **per-patch** embeddings that
are individually meaningful, not just a single pooled image vector. PaliGemma feeds all 256
of them to Gemma as 256 tokens, so the language model can attend to *parts* of the image.
That is why our `SiglipVisionModel` has no pooling head and no `[CLS]` token.

## SigLIP vs CLIP, summarised

| | CLIP | SigLIP |
|---|---|---|
| Loss | softmax cross-entropy, both directions | pairwise binary cross-entropy |
| Needs the full batch matrix | yes | no |
| Learnable params in the loss | temperature | scale `t'` **and** bias `b` |
| Batch-size dependence | wants it huge | works well small, scales to 1 M |
| Memory per device | `O(N²)` | `O(chunk²)` |

## Run it

```bash
cd 04-siglip-sigmoid-loss/code
python siglip_loss.py
```

The interesting part is section 3: the block-wise sum matching the full loss to `0.000e+00`.
Try to write the equivalent block-wise computation for the CLIP loss and you will feel
exactly where it becomes impossible.

## Key takeaways

- Sigmoid on each pair instead of softmax over the batch.
- The bias `b ≈ −10` counteracts the N:N² positive/negative imbalance.
- The loss is decomposable → shardable → million-scale batches, and better small-batch
  behaviour.
- We want this encoder because contrastive training makes its embeddings
  language-aligned, and because it gives us 256 meaningful per-patch tokens.

## Gotchas

- **Sum vs mean.** The paper divides the summed pair losses by N (not N²). Get this wrong
  and your learning rate is off by a factor of the batch size.
- **`logsigmoid`, not `log(sigmoid(x))`** — same reasoning as Episode 03.
- **SigLIP the loss vs SigLIP the model.** From here on, "SigLIP" means the vision tower we
  are about to build. The loss never appears in our code — we only do inference.

## Exercises

1. Set `b = 0` and plot the loss as N grows from 8 to 512 on random embeddings. Explain the
   trend in terms of the positive/negative ratio.
2. Implement the block-wise loss for a "ring" of 4 devices where each device only ever
   holds 1/4 of the text embeddings at a time. Confirm it still matches the full loss.
3. Try to write the CLIP loss block-wise. What exactly do you need to communicate between
   blocks, and how many passes over the data does it take?
4. The final SigLIP checkpoint is trained at 224×224 with `patch_size=14`. Compute how many
   tokens per image that is, and check it against `paligemma-3b-pt-224`'s 256.

## Further reading

- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP),
  <https://arxiv.org/abs/2303.15343>
- Beyer et al., *PaliGemma: A versatile 3B VLM for transfer*,
  <https://arxiv.org/abs/2407.07726>

**Previous:** [Episode 03](../03-numerical-stability-of-softmax/) ·
**Next:** [Episode 05 — The Vision Transformer](../05-vision-transformer/)
