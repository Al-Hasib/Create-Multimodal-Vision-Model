# Episode 01 — Introduction & Setup

> **Video chapter:** `00:00:00 – 00:05:52` (Introduction)
> **Target runtime:** 8–10 min · **Code:** [`code/check_env.py`](code/check_env.py)

## What this playlist builds

One model, PaliGemma, written from scratch in PyTorch: **image + text in, text out.**
No `transformers` model classes, no `nn.MultiheadAttention`, no `nn.TransformerEncoder`.
By Episode 20 you load Google's real 3-billion-parameter checkpoint into *your* code and
it captions an image.

```
                       "this building is"
                              |
   image ──► SigLIP vision ──► linear ──►┐
             encoder (ViT)    projector  │
                                         ▼
                              [img tokens][text tokens] ──► Gemma ──► "the Eiffel Tower"
                                                            decoder
```

Three components, built in this order:

| Part | What it is | Episodes |
|---|---|---|
| **SigLIP** | a Vision Transformer that turns an image into 256 embeddings | 02–10 |
| **Projector** | one `nn.Linear` that maps vision dims → language dims | 11–13 |
| **Gemma** | a decoder-only language model (RMSNorm, GQA, RoPE, KV-cache) | 14–19 |
| **Inference** | prefill/decode loop, top-p sampling | 20 |

## Why write it by hand

You can call `PaliGemmaForConditionalGeneration.from_pretrained` in one line. That line
teaches you nothing about *why* there is a `sqrt(hidden_size)` on the image features, why
the prompt is not causally masked, or why `weight` in RMSNorm is initialised to zeros.
Every one of those details is a real bug you will hit when you adapt a model, and each
gets its own section in this playlist.

## Prerequisites

**Required:** Python, and enough PyTorch to know what a `Tensor` and an `nn.Module` are.
**Assumed known:** matrix multiplication, softmax, gradient descent, cross-entropy.
**Explained from scratch, do not worry if these are new:** the transformer architecture,
attention, normalization layers, positional encodings, KV-caching, sampling.

You do **not** need a GPU. Every demo in this repository runs in under two seconds on a
laptop CPU, and the real 3B model generates on CPU too (slowly — minutes, not seconds).

## Setup

```bash
git clone <this-repo>
cd Create-Multimodal-Vision-Model
pip install -r requirements.txt
python 01-introduction-and-setup/code/check_env.py
```

`check_env.py` prints your Python and package versions, the device that will be used, and
exits non-zero if something is missing. `requirements.txt` pins the versions this code was
written against; newer PyTorch works fine.

## The weights (do this now — it downloads ~6 GB)

You only need them in Episode 20, but the license step takes a few minutes of waiting.

1. Create a Hugging Face account and open <https://huggingface.co/google/paligemma-3b-pt-224>.
2. Accept Google's usage license (it is a gated repository — the download 401s otherwise).
3. `huggingface-cli login`
4. Download:

```bash
huggingface-cli download google/paligemma-3b-pt-224 \
    --local-dir ~/projects/paligemma-weights/paligemma-3b-pt-224
```

`paligemma-3b-pt-224` is the base pretrained model at 224×224 resolution — the smallest
and the one whose config we follow throughout. `pt` = pretrained (not instruction tuned).

## How this repository is organised

```
01-introduction-and-setup/     ← you are here
  README.md                    ← the lesson: theory, walkthrough, exercises
  code/                        ← runnable code for this episode
...
20-inference-and-top-p-sampling/
modeling_siglip.py             ← THE FINAL CODE lives at the repo root
modeling_gemma.py
processing_paligemma.py
inference.py
utils.py
tools/build_snapshots.py       ← generates each episode's code/ from the final files
```

Two rules that make the playlist work:

- **Each `NN-*/code/` folder is self-contained and runnable.** It holds the model files as
  they look at the *end* of that episode. Anything not yet taught is an explicit stub
  marked `PLACEHOLDER -- BUILT IN EPISODE NN`, so the file always imports and runs.
- **`git diff` between two episodes' `code/` folders is exactly what that episode added.**
  Great for review, and great for catching a typo when your version misbehaves.

```bash
# what did Episode 10 actually change?
diff 08-siglip-encoder-and-ffn/code/modeling_siglip.py \
     10-multi-head-attention-code/code/modeling_siglip.py
```

If you change the final files at the root, regenerate the snapshots:

```bash
python tools/build_snapshots.py           # rewrite them
python tools/build_snapshots.py --check    # CI-friendly staleness check
```

## How to follow along

Type the code. Do not copy-paste it. Then run the episode's `check_*.py` / `*_demo.py`
script and read the numbers — every one of them was chosen to make a specific claim
falsifiable. Several of these scripts intentionally demonstrate what the model *cannot*
do yet (Episode 08 is the clearest example).

## Episode roadmap

| # | Episode | Chapter |
|---|---|---|
| 01 | Introduction & setup | `00:00:00` |
| 02 | Contrastive learning & CLIP | `00:05:52` |
| 03 | Numerical stability of the softmax | `00:16:50` |
| 04 | SigLIP: the sigmoid loss | `00:23:00` |
| 05 | The Vision Transformer | `00:29:13` |
| 06 | Coding SigLIP: config & embeddings | `00:35:38` |
| 07 | Batch norm vs layer norm | `00:54:25` |
| 08 | The SigLIP encoder & FFN | `01:05:28` |
| 09 | Multi-head attention: theory | `01:20:45` |
| 10 | Multi-head attention: code | `01:50:00` |
| 11 | PaliGemma architecture & input processor | `02:18:30` |
| 12 | Gemma config & weight tying | `02:40:56` |
| 13 | Merging image & text embeddings | `02:46:20` |
| 14 | The KV-cache | `03:08:54` |
| 15 | Attention mask & position ids | `03:33:35` |
| 16 | GemmaModel & RMS normalization | `03:53:17` |
| 17 | Decoder layer & GeGLU FFN | `04:09:50` |
| 18 | Grouped-query attention | `04:16:02` |
| 19 | Rotary positional embedding | `04:56:00` |
| 20 | Inference & top-p sampling | `05:23:40` |

## Credits

The curriculum follows Umar Jamil's *Coding a Multimodal (Vision) Language Model from
scratch in PyTorch with full explanation* and the reference implementation
[`hkproj/pytorch-paligemma`](https://github.com/hkproj/pytorch-paligemma). The model is
Google's PaliGemma ([paper](https://arxiv.org/abs/2407.07726)).

**Next:** [Episode 02 — Contrastive learning & CLIP](../02-contrastive-learning-and-clip/)
