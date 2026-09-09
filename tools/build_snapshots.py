"""Build the per-episode code snapshots of the playlist from the final source files.

Every coding episode ships a *runnable* snapshot of `modeling_siglip.py` /
`modeling_gemma.py` as they look at the END of that episode. Instead of keeping ten
hand-maintained copies of the same file in sync, we generate them here: the blocks
(top-level classes / functions) that have already been taught are copied verbatim
from the final implementation at the repository root, and the blocks that only
arrive in a later episode are replaced by an explicit TODO stub, so every snapshot
still imports and runs.

Usage:
    python tools/build_snapshots.py           # regenerate every snapshot
    python tools/build_snapshots.py --check   # fail if a snapshot is out of date
"""

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENERATED_HEADER = """# ---------------------------------------------------------------------------
# GENERATED FILE -- do not edit by hand.
#
# Snapshot of `{source}` as it looks at the end of
# Episode {episode_no}: {episode_title}.
#
# Regenerate with:  python tools/build_snapshots.py
# Everything marked "PLACEHOLDER -- BUILT IN EPISODE ..." is filled in later, so
# this file always imports and runs as-is.
# ---------------------------------------------------------------------------
"""


def parse_blocks(path: Path):
    """Return (imports_source, {block_name: block_source})."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    imports, blocks = [], {}
    for node in tree.body:
        segment = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(segment)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            blocks[node.name] = segment
        else:  # module level constants
            blocks[segment] = segment
    return "\n".join(imports), blocks


# ---------------------------------------------------------------------------
# Placeholders: what a not-yet-taught block looks like inside a snapshot.
# ---------------------------------------------------------------------------

STUB_SIGLIP_ATTENTION = '''
class SiglipAttention(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 10 (Multi-Head Attention).

    The encoder layer below already calls it, so for now we return the input
    unchanged. That makes the encoder a residual + LayerNorm + MLP stack: the
    shapes are right end to end, only the mixing between patches is missing.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

    def forward(self, hidden_states: torch.Tensor):
        # TODO(Episode 10): project to Q, K, V and compute softmax(Q @ K^T / sqrt(d_k)) @ V.
        return hidden_states, None
'''

STUB_KV_CACHE = '''
class KVCache():
    """PLACEHOLDER -- BUILT IN EPISODE 14 (KV-Cache).

    It exists so that the `Optional[KVCache]` type hints below resolve.
    """

    def num_items(self) -> int:
        return 0
'''

STUB_GEMMA_MODEL = '''
class GemmaModel(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 16 (GemmaModel & RMSNorm).

    For now it is only the token embedding table plus a pass-through forward,
    which is all we need to embed the prompt and merge in the image features.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        kv_cache=None,
    ) -> torch.FloatTensor:
        # TODO(Episode 16): the normalizer, the stack of decoder layers, the final RMSNorm.
        return inputs_embeds
'''

STUB_GEMMA_DECODER_LAYER = '''
class GemmaDecoderLayer(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 17 (decoder layer & GeGLU FFN)."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None):
        # TODO(Episode 17): RMSNorm -> attention -> residual -> RMSNorm -> MLP -> residual.
        return hidden_states
'''

STUB_GEMMA_ATTENTION = '''
class GemmaAttention(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 18 (Grouped-Query Attention)."""

    def __init__(self, config: GemmaConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask=None, position_ids=None, kv_cache=None, **kwargs):
        # TODO(Episode 18): Q/K/V projections, RoPE, grouped-query attention, mask, softmax, o_proj.
        return hidden_states, None
'''

STUB_ROTARY_EMBEDDING = '''
class GemmaRotaryEmbedding(nn.Module):
    """PLACEHOLDER -- BUILT IN EPISODE 19 (Rotary Positional Embedding).

    Returns cos = 1 and sin = 0, which makes `apply_rotary_pos_emb` the identity:
    attention runs, but the model is position-blind (a bag of tokens).
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        batch_size, seq_len = position_ids.shape
        shape = (batch_size, seq_len, self.dim)
        # TODO(Episode 19): theta_i = base ** (-2i / dim), then cos/sin of (position * theta).
        cos = torch.ones(shape, dtype=x.dtype, device=x.device)
        sin = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return cos, sin
'''

STUB_APPLY_ROTARY = '''
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """PLACEHOLDER -- BUILT IN EPISODE 19 (Rotary Positional Embedding)."""
    # TODO(Episode 19): return q * cos + rotate_half(q) * sin, and the same for k.
    return q, k
'''

STUB_PALIGEMMA_12 = '''
class PaliGemmaForConditionalGeneration(nn.Module):
    """The top level model: vision tower + projector + language model.

    In this episode we only wire the three submodules together and tie the weights.
    The forward pass is built over the next episodes.
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        raise NotImplementedError("BUILT IN EPISODE 13 (merging image and text embeddings).")
'''

# Episode 13 builds only the first half of the merge method: the embeddings.
# The attention mask and the position ids arrive in Episode 15.
STUB_PALIGEMMA_13 = '''
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # TODO(Episode 15): build the attention mask and the position ids here.
        return final_embedding, None, None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        raise NotImplementedError("BUILT IN EPISODE 15 (attention mask, position ids, projection).")
'''

STUBS = {
    "SiglipAttention": STUB_SIGLIP_ATTENTION,
    "KVCache": STUB_KV_CACHE,
    "GemmaModel": STUB_GEMMA_MODEL,
    "GemmaDecoderLayer": STUB_GEMMA_DECODER_LAYER,
    "GemmaAttention": STUB_GEMMA_ATTENTION,
    "GemmaRotaryEmbedding": STUB_ROTARY_EMBEDDING,
    "apply_rotary_pos_emb": STUB_APPLY_ROTARY,
    "PaliGemmaForConditionalGeneration@12": STUB_PALIGEMMA_12,
    "PaliGemmaForConditionalGeneration@13": STUB_PALIGEMMA_13,
}


# ---------------------------------------------------------------------------
# The recipes: which blocks exist at the end of each episode.
# A plain string is copied verbatim from the final file, ("stub", key) is a
# placeholder from above.
# ---------------------------------------------------------------------------

SIGLIP_ALL = [
    "SiglipVisionConfig",
    "SiglipVisionEmbeddings",
    "SiglipAttention",
    "SiglipMLP",
    "SiglipEncoderLayer",
    "SiglipEncoder",
    "SiglipVisionTransformer",
    "SiglipVisionModel",
]

GEMMA_ALL = [
    "KVCache",
    "GemmaConfig",
    "PaliGemmaConfig",
    "GemmaRMSNorm",
    "GemmaRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "GemmaMLP",
    "repeat_kv",
    "GemmaAttention",
    "GemmaDecoderLayer",
    "GemmaModel",
    "GemmaForCausalLM",
    "PaliGemmaMultiModalProjector",
    "PaliGemmaForConditionalGeneration",
]

EPISODES = [
    (
        "06-coding-siglip-embeddings",
        6,
        "Coding SigLip - config and vision embeddings",
        {"modeling_siglip.py": ["SiglipVisionConfig", "SiglipVisionEmbeddings"]},
    ),
    (
        "08-siglip-encoder-and-ffn",
        8,
        "Coding SigLip - the encoder layer and the FFN",
        {
            "modeling_siglip.py": [
                "SiglipVisionConfig",
                "SiglipVisionEmbeddings",
                ("stub", "SiglipAttention"),
                "SiglipMLP",
                "SiglipEncoderLayer",
                "SiglipEncoder",
                "SiglipVisionTransformer",
                "SiglipVisionModel",
            ]
        },
    ),
    (
        "10-multi-head-attention-code",
        10,
        "Coding Multi-Head Attention, finishing SigLip",
        {"modeling_siglip.py": SIGLIP_ALL},
    ),
    (
        "12-gemma-config-weight-tying",
        12,
        "Gemma configs, weight tying and the PaliGemma skeleton",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                ("stub", "KVCache"),
                "GemmaConfig",
                "PaliGemmaConfig",
                ("stub", "GemmaModel"),
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                ("stub", "PaliGemmaForConditionalGeneration@12"),
            ],
        },
    ),
    (
        "13-merging-image-and-text-embeddings",
        13,
        "Merging image and text embeddings",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                ("stub", "KVCache"),
                "GemmaConfig",
                "PaliGemmaConfig",
                ("stub", "GemmaModel"),
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                ("stub", "PaliGemmaForConditionalGeneration@13"),
            ],
        },
    ),
    (
        "15-attention-mask-and-position-ids",
        15,
        "The attention mask, the position ids and the image projection",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                "KVCache",
                "GemmaConfig",
                "PaliGemmaConfig",
                ("stub", "GemmaModel"),
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                "PaliGemmaForConditionalGeneration",
            ],
        },
    ),
    (
        "16-gemma-model-and-rms-norm",
        16,
        "GemmaModel, RMS normalization and the LM head",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                "KVCache",
                "GemmaConfig",
                "PaliGemmaConfig",
                "GemmaRMSNorm",
                ("stub", "GemmaDecoderLayer"),
                "GemmaModel",
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                "PaliGemmaForConditionalGeneration",
            ],
        },
    ),
    (
        "17-decoder-layer-and-geglu-ffn",
        17,
        "The Gemma decoder layer and the GeGLU FFN",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                "KVCache",
                "GemmaConfig",
                "PaliGemmaConfig",
                "GemmaRMSNorm",
                "GemmaMLP",
                ("stub", "GemmaAttention"),
                "GemmaDecoderLayer",
                "GemmaModel",
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                "PaliGemmaForConditionalGeneration",
            ],
        },
    ),
    (
        "18-grouped-query-attention",
        18,
        "Grouped-Query Attention and the Gemma attention block",
        {
            "modeling_siglip.py": SIGLIP_ALL,
            "modeling_gemma.py": [
                "KVCache",
                "GemmaConfig",
                "PaliGemmaConfig",
                "GemmaRMSNorm",
                ("stub", "GemmaRotaryEmbedding"),
                ("stub", "apply_rotary_pos_emb"),
                "GemmaMLP",
                "repeat_kv",
                "GemmaAttention",
                "GemmaDecoderLayer",
                "GemmaModel",
                "GemmaForCausalLM",
                "PaliGemmaMultiModalProjector",
                "PaliGemmaForConditionalGeneration",
            ],
        },
    ),
    (
        "19-rotary-positional-embedding",
        19,
        "Rotary Positional Embedding",
        {"modeling_siglip.py": SIGLIP_ALL, "modeling_gemma.py": GEMMA_ALL},
    ),
    (
        "20-inference-and-top-p-sampling",
        20,
        "Inference, Top-P sampling and running the model",
        {"modeling_siglip.py": SIGLIP_ALL, "modeling_gemma.py": GEMMA_ALL},
    ),
]

# Files copied verbatim (nothing progressive to build) into an episode's code folder.
VERBATIM = {
    "11-paligemma-input-processor": ["processing_paligemma.py"],
    "12-gemma-config-weight-tying": ["processing_paligemma.py"],
    "13-merging-image-and-text-embeddings": ["processing_paligemma.py"],
    "15-attention-mask-and-position-ids": ["processing_paligemma.py"],
    "16-gemma-model-and-rms-norm": ["processing_paligemma.py"],
    "17-decoder-layer-and-geglu-ffn": ["processing_paligemma.py"],
    "18-grouped-query-attention": ["processing_paligemma.py"],
    "19-rotary-positional-embedding": ["processing_paligemma.py"],
    "20-inference-and-top-p-sampling": [
        "processing_paligemma.py",
        "inference.py",
        "utils.py",
        "launch_inference.sh",
    ],
}


def render(source_name: str, episode_no: int, episode_title: str, recipe) -> str:
    imports, blocks = parse_blocks(ROOT / source_name)
    parts = [
        GENERATED_HEADER.format(
            source=source_name, episode_no=f"{episode_no:02d}", episode_title=episode_title
        ),
        imports,
    ]
    for item in recipe:
        if isinstance(item, str):
            if item not in blocks:
                raise SystemExit(f"{source_name}: no top-level block named {item!r}")
            body = blocks[item]
        else:
            body = STUBS[item[1]]
        parts.append("\n" + body.strip() + "\n")
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify the snapshots are up to date")
    args = parser.parse_args()

    stale, written = [], 0
    targets = []

    for folder, episode_no, episode_title, sources in EPISODES:
        for source_name, recipe in sources.items():
            targets.append(
                (ROOT / folder / "code" / source_name, render(source_name, episode_no, episode_title, recipe))
            )

    for folder, names in VERBATIM.items():
        for name in names:
            targets.append((ROOT / folder / "code" / name, (ROOT / name).read_text(encoding="utf-8")))

    for target, content in targets:
        if args.check:
            if not target.exists() or target.read_text(encoding="utf-8") != content:
                stale.append(str(target.relative_to(ROOT)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written += 1

    if args.check:
        if stale:
            print("Out of date snapshots:")
            for path in stale:
                print("  " + path)
            return 1
        print(f"All {len(targets)} snapshots are up to date.")
        return 0

    print(f"Wrote {written} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
