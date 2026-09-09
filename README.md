# Build a Multimodal Vision-Language Model from Scratch

A 20-episode YouTube course that builds **PaliGemma** — Google's 3B vision-language model —
from an empty file in PyTorch. Image + text in, text out. No `transformers` model classes,
no `nn.MultiheadAttention`, no `nn.TransformerEncoder`. By the last episode you load
Google's real checkpoint into your own code and it captions an image.

```
                       "this building is"
                              |
   image ──► SigLIP vision ──► linear ──►┐
             encoder (ViT)    projector  │
                                         ▼
                              [img tokens][text tokens] ──► Gemma ──► "the Eiffel Tower"
                                                            decoder
```

## Start here

```bash
pip install -r requirements.txt
python 01-introduction-and-setup/code/check_env.py
```

Then open [`01-introduction-and-setup/`](01-introduction-and-setup/) and work forward.
Each episode folder holds one lesson: a `README.md` with the full teaching notes and a
`code/` folder you can run.

## The episodes

| # | Episode | Chapter | Builds |
|---|---|---|---|
| 01 | [Introduction & setup](01-introduction-and-setup/) | `00:00:00` | environment check |
| 02 | [Contrastive learning & CLIP](02-contrastive-learning-and-clip/) | `00:05:52` | the CLIP loss, by hand |
| 03 | [Numerical stability of the softmax](03-numerical-stability-of-softmax/) | `00:16:50` | stable softmax, log-sum-exp |
| 04 | [SigLIP: the sigmoid loss](04-siglip-sigmoid-loss/) | `00:23:00` | sigmoid loss, sharding proof |
| 05 | [The Vision Transformer](05-vision-transformer/) | `00:29:13` | patch embedding equivalence |
| 06 | [Coding SigLIP: config & embeddings](06-coding-siglip-embeddings/) | `00:35:38` | `SiglipVisionConfig`, `SiglipVisionEmbeddings` |
| 07 | [Batch norm vs layer norm](07-batch-norm-vs-layer-norm/) | `00:54:25` | both norms from scratch |
| 08 | [The SigLIP encoder & FFN](08-siglip-encoder-and-ffn/) | `01:05:28` | `SiglipMLP`, `EncoderLayer`, `Encoder`, `VisionModel` |
| 09 | [Multi-head attention: theory](09-multi-head-attention-theory/) | `01:20:45` | attention by hand |
| 10 | [Multi-head attention: code](10-multi-head-attention-code/) | `01:50:00` | `SiglipAttention` — **vision tower done** |
| 11 | [PaliGemma architecture & input processor](11-paligemma-input-processor/) | `02:18:30` | `processing_paligemma.py` |
| 12 | [Gemma config & weight tying](12-gemma-config-weight-tying/) | `02:40:56` | `GemmaConfig`, `PaliGemmaConfig`, projector |
| 13 | [Merging image & text embeddings](13-merging-image-and-text-embeddings/) | `02:46:20` | `_merge_input_ids_with_image_features` |
| 14 | [The KV-cache](14-kv-cache/) | `03:08:54` | cache theory, 17× measured |
| 15 | [Attention mask & position ids](15-attention-mask-and-position-ids/) | `03:33:35` | `KVCache`, the prefix-LM mask |
| 16 | [GemmaModel & RMS normalization](16-gemma-model-and-rms-norm/) | `03:53:17` | `GemmaModel`, `GemmaRMSNorm`, LM head |
| 17 | [Decoder layer & GeGLU FFN](17-decoder-layer-and-geglu-ffn/) | `04:09:50` | `GemmaDecoderLayer`, `GemmaMLP` |
| 18 | [Grouped-query attention](18-grouped-query-attention/) | `04:16:02` | `repeat_kv`, `GemmaAttention` |
| 19 | [Rotary positional embedding](19-rotary-positional-embedding/) | `04:56:00` | `GemmaRotaryEmbedding` — **model done** |
| 20 | [Inference & top-p sampling](20-inference-and-top-p-sampling/) | `05:23:40` | `utils.py`, `inference.py`, run it |

See [CURRICULUM.md](CURRICULUM.md) for the chapter-to-episode mapping, the concept
dependency graph, and per-episode runtimes.

## How the repository is laid out

```
modeling_siglip.py          THE FINAL CODE — the vision tower
modeling_gemma.py           the language model + the multimodal glue
processing_paligemma.py     image and prompt preprocessing
inference.py, utils.py      weight loading and the generation loop
launch_inference.sh

01-introduction-and-setup/
  README.md                 the lesson
  code/                     runnable code for this episode
...
20-inference-and-top-p-sampling/

tools/build_snapshots.py    generates every episode's code/ from the final files
notes/                      slides (untouched by the course material)
```

**Each `NN-*/code/` folder is self-contained and runnable.** It contains the model files as
they look at the *end* of that episode; anything not yet taught is an explicit stub marked
`PLACEHOLDER -- BUILT IN EPISODE NN`, so every snapshot imports and runs.

That makes the diff between two episodes exactly what the later one added:

```bash
diff 08-siglip-encoder-and-ffn/code/modeling_siglip.py \
     10-multi-head-attention-code/code/modeling_siglip.py     # what attention added
```

The snapshots are generated, not hand-maintained. If you edit the final files at the root:

```bash
python tools/build_snapshots.py            # regenerate
python tools/build_snapshots.py --check    # CI-friendly staleness check
```

## Running everything

```bash
# every demo and check in the course (a few seconds each, CPU only)
for d in */code; do (cd "$d" && for f in *.py; do
  case "$f" in modeling_*|processing_*|inference.py|utils.py) continue;; esac
  python "$f" >/dev/null && echo "ok   $d/$f" || echo "FAIL $d/$f"; done); done
```

All 22 should pass — except `01-.../check_env.py`, which exits non-zero on purpose if a
package from `requirements.txt` is missing.

Only Episode 20's `launch_inference.sh` needs the downloaded weights.

## What you need

- Python 3.9+ and PyTorch. **No GPU required** — every demo runs on CPU in seconds, and the
  real 3B model generates on CPU too (minutes, not seconds).
- For Episode 20: a Hugging Face account, acceptance of Google's license, and ~6 GB for
  [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224).

Prerequisite knowledge: Python, basic PyTorch, matrix multiplication, softmax, gradient
descent. The transformer itself, attention, normalization, positional encodings, KV-caching
and sampling are all built from scratch.

## What the course covers

Concepts, not just code: contrastive learning (CLIP and SigLIP), numerical stability of the
softmax, vision transformers and patch embeddings, batch vs layer vs RMS normalization,
pre-norm residual blocks, multi-head attention, prefix-LM masking, KV-caching, multi-query
and grouped-query attention, rotary positional embeddings, GeGLU feed-forward networks,
weight tying, and top-p sampling.

Every claim in the notes is backed by a script that measures it. Several episodes
deliberately demonstrate what the model *cannot* do yet — Episode 08 proves that without
attention no patch influences any other, and Episode 18 leaves the model position-blind
until Episode 19 fixes it.

## Credits

Curriculum and code follow Umar Jamil's *Coding a Multimodal (Vision) Language Model from
scratch in PyTorch with full explanation* and the reference implementation
[`hkproj/pytorch-paligemma`](https://github.com/hkproj/pytorch-paligemma).

Papers: [PaliGemma](https://arxiv.org/abs/2407.07726) ·
[SigLIP](https://arxiv.org/abs/2303.15343) · [CLIP](https://arxiv.org/abs/2103.00020) ·
[ViT](https://arxiv.org/abs/2010.11929) · [Gemma](https://arxiv.org/abs/2403.08295) ·
[RoPE](https://arxiv.org/abs/2104.09864) · [GQA](https://arxiv.org/abs/2305.13245) ·
[RMSNorm](https://arxiv.org/abs/1910.07467) · [GLU variants](https://arxiv.org/abs/2002.05202)
