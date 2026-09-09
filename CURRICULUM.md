# Curriculum map

How the 30 video chapters map onto 20 episodes, what each episode adds to the codebase, and
which concepts depend on which.

## Chapter → episode mapping

| Video chapter | Timestamp | Episode |
|---|---|---|
| Introduction | `00:00:00` | 01 |
| Contrastive Learning and CLIP | `00:05:52` | 02 |
| Numerical stability of the Softmax | `00:16:50` | 03 |
| SigLip | `00:23:00` | 04 |
| Why a Contrastive Vision Encoder? | `00:26:30` | 04 |
| Vision Transformer | `00:29:13` | 05 |
| Coding SigLip | `00:35:38` | 06 |
| Batch Normalization, Layer Normalization | `00:54:25` | 07 |
| Coding SigLip (Encoder) | `01:05:28` | 08 |
| Coding SigLip (FFN) | `01:16:12` | 08 |
| Multi-Head Attention (Coding + Explanation) | `01:20:45` | 09 (theory) + 10 (code) |
| Coding SigLip | `02:15:40` | 10 |
| PaliGemma Architecture review | `02:18:30` | 11 |
| PaliGemma input processor | `02:21:19` | 11 |
| Coding Gemma | `02:40:56` | 12 |
| Weight tying | `02:43:44` | 12 |
| Coding Gemma | `02:46:20` | 13 |
| KV-Cache (Explanation) | `03:08:54` | 14 |
| Coding Gemma | `03:33:35` | 15 |
| Image features projection | `03:52:05` | 15 |
| Coding Gemma | `03:53:17` | 16 |
| RMS Normalization | `04:02:45` | 16 |
| Gemma Decoder Layer | `04:09:50` | 17 |
| Gemma FFN (MLP) | `04:12:44` | 17 |
| Multi-Head Attention (Coding) | `04:16:02` | 18 |
| Grouped Query Attention | `04:18:30` | 18 |
| Multi-Head Attention (Coding) | `04:38:35` | 18 |
| KV-Cache (Coding) | `04:43:26` | 18 |
| Multi-Head Attention (Coding) | `04:47:44` | 18 |
| Rotary Positional Embedding | `04:56:00` | 19 |
| Inference code | `05:23:40` | 20 |
| Top-P Sampling | `05:32:50` | 20 |
| Inference code | `05:40:40` | 20 |
| Conclusion | `05:43:40` | 20 |

Two editorial decisions:

- **The 55-minute attention chapter is split in two** (Episodes 09 and 10) at roughly
  `01:50:00`: theory first, then code. It is the longest single chapter in the video and the
  most important idea in the playlist.
- **The five interleaved chapters between `04:16:02` and `04:56:00` are one episode** (18).
  They are one continuous task — writing `GemmaAttention` — with GQA and the cache explained
  as they come up.

## What each episode adds to the codebase

| Ep | File | Added |
|---|---|---|
| 06 | `modeling_siglip.py` | `SiglipVisionConfig`, `SiglipVisionEmbeddings` |
| 08 | `modeling_siglip.py` | `SiglipMLP`, `SiglipEncoderLayer`, `SiglipEncoder`, `SiglipVisionTransformer`, `SiglipVisionModel` (attention stubbed) |
| 10 | `modeling_siglip.py` | `SiglipAttention` → **file complete** |
| 11 | `processing_paligemma.py` | `rescale`, `resize`, `normalize`, `process_images`, `add_image_tokens_to_prompt`, `PaliGemmaProcessor` → **file complete** |
| 12 | `modeling_gemma.py` | `GemmaConfig`, `PaliGemmaConfig`, `PaliGemmaMultiModalProjector`, `GemmaForCausalLM`, `tie_weights` |
| 13 | `modeling_gemma.py` | `_merge_input_ids_with_image_features` (embeddings half) |
| 15 | `modeling_gemma.py` | `KVCache`, the mask + position ids, `PaliGemmaForConditionalGeneration.forward` |
| 16 | `modeling_gemma.py` | `GemmaRMSNorm`, `GemmaModel` |
| 17 | `modeling_gemma.py` | `GemmaMLP`, `GemmaDecoderLayer` |
| 18 | `modeling_gemma.py` | `repeat_kv`, `GemmaAttention` |
| 19 | `modeling_gemma.py` | `GemmaRotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb` → **file complete** |
| 20 | `utils.py`, `inference.py` | `load_hf_model`, generation loop, `_sample_top_p` |

Episodes 01–05, 07, 09 and 14 are theory: they ship a self-contained demo script instead of
model code.

## The stubs, and where they are filled in

Each coding episode's snapshot runs even though later parts are missing. The placeholders are
part of the pedagogy — Episode 08's stubbed attention is what makes Episode 10's measurement
meaningful.

| Stub | Present in | Filled in |
|---|---|---|
| `SiglipAttention` returns its input | 08 | 10 |
| `KVCache` with `num_items() == 0` | 12–13 | 15 |
| `GemmaModel` is embed + pass-through | 12–15 | 16 |
| `GemmaDecoderLayer` is identity | 16 | 17 |
| `GemmaAttention` is identity | 17 | 18 |
| `GemmaRotaryEmbedding` returns cos=1, sin=0 | 18 | 19 |
| `PaliGemma.forward` raises | 12 | 13 (partial), 15 (complete) |

## Concept dependencies

```
02 contrastive/CLIP ──┬──► 04 SigLIP loss ──► 06 coding the tower
03 softmax stability ─┘         │                     ▲
                                │              05 ViT / patches
                                ▼                     │
                        09 attention theory ──► 10 attention code
                                ▲                     │
                       07 norms ┘                     ▼
                                              08 encoder + FFN
                                                      │
                                                      ▼
                                        11 architecture + processor
                                                      │
                                                      ▼
                                        12 configs + weight tying
                                                      │
                                                      ▼
                                        13 merging embeddings ◄── 16 (the sqrt scale-up)
                                                      │
                                    14 KV-cache ──► 15 mask + positions
                                                      │
                                        16 GemmaModel + RMSNorm
                                                      │
                                        17 decoder layer + GeGLU
                                                      │
                          14 (cache) ──────► 18 GQA + attention
                                                      │
                                        19 RoPE  ◄── 15 (position ids)
                                                      │
                                        20 inference + sampling
```

Three cross-episode pairs worth flagging when teaching, because neither half makes sense
alone:

- **13 ↔ 16** — the `/ sqrt(hidden_size)` on image features exists only to cancel
  `GemmaModel`'s `* sqrt(hidden_size)`.
- **14 ↔ 18** — the KV-cache motivates grouped-query attention; GQA is what makes the cache
  affordable.
- **15 ↔ 19** — `position_ids` are built in 15 and consumed by RoPE in 19; nothing in
  between uses them.

## Runtimes

| Ep | Minutes | | Ep | Minutes |
|---|---|---|---|---|
| 01 | 8–10 | | 11 | 20–22 |
| 02 | 15–18 | | 12 | 15–18 |
| 03 | 10–12 | | 13 | 20–22 |
| 04 | 12–15 | | 14 | 22–25 |
| 05 | 12–15 | | 15 | 18–20 |
| 06 | 18–20 | | 16 | 16–18 |
| 07 | 12–15 | | 17 | 10–12 |
| 08 | 15–18 | | 18 | 25–30 |
| 09 | 25–30 | | 19 | 28–30 |
| 10 | 20–25 | | 20 | 20–25 |

Roughly 5.5–6.5 hours total, matching the source video, in pieces that survive being watched
on separate days.

## Suggested recording order

Not the same as the release order. Record **06, 08, 10** (the SigLIP code) as one session
while the file is in your head, then insert 05, 07 and 09 in front of them. Same for
**12–19**: the Gemma file is one continuous build, and 14's theory is easier to explain after
you have written the cache. Theory episodes are independent and can be recorded any time.

## Per-episode checklist for recording

Each episode README already carries all six sections in this order:

1. the chapter reference and runtime
2. the concept, with the failure it solves
3. the code, with shape comments
4. what to run, and what the output should say
5. key takeaways
6. gotchas and exercises

The **gotchas** are the most valuable part to say out loud on camera: nearly all of them are
bugs that run without error and produce plausible output — silent module renames, the
RMSNorm `1.0 +`, interleaved `repeat_kv`, caching before RoPE, a missing `gather` in top-p.
