"""Episode 16 -- GemmaModel: the embedding scale-up, the layer stack, the LM head.

Run:  python check_gemma_model.py
"""

import torch

from modeling_gemma import GemmaConfig, GemmaForCausalLM, KVCache

torch.manual_seed(0)

CONFIG = dict(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
)


def main() -> None:
    config = GemmaConfig(**CONFIG, pad_token_id=0)
    model = GemmaForCausalLM(config).eval()
    model.tie_weights()

    print("=" * 74)
    print("1. THE STRUCTURE")
    print("=" * 74)
    print("  GemmaForCausalLM")
    print("    .model      = GemmaModel")
    print("        .embed_tokens : Embedding [Vocab_Size, Hidden_Size]")
    print(f"        .layers       : {config.num_hidden_layers} x GemmaDecoderLayer"
          "   (still a stub, Episode 17)")
    print("        .norm         : GemmaRMSNorm")
    print("    .lm_head    = Linear(Hidden_Size -> Vocab_Size, bias=False), tied to embed_tokens")
    print("\n  The split matters: GemmaModel produces hidden states, GemmaForCausalLM")
    print("  adds the head that turns them into vocabulary logits. Swap the head and")
    print("  the same body does classification or regression instead.")

    print()
    print("=" * 74)
    print("2. THE EMBEDDING NORMALIZER")
    print("=" * 74)
    input_ids = torch.tensor([[5, 6, 7]])
    embeds = model.get_input_embeddings()(input_ids)
    normalizer = config.hidden_size**0.5
    print(f"  hidden_states = inputs_embeds * sqrt(hidden_size) = * sqrt({config.hidden_size}) = * {normalizer:.1f}")
    print(f"  ||embedding||          = {embeds[0, 0].norm():.4f}")
    print(f"  ||embedding * {normalizer:.0f}||     = {(embeds[0, 0] * normalizer).norm():.4f}")
    print("\n  Two consequences worth internalising:")
    print("    - this is why Episode 13 pre-divided the image features by the same")
    print("      sqrt(hidden_size): they must not be scaled up twice")
    print("    - it is computed as a tensor in the hidden_states dtype on purpose;")
    print("      in bfloat16 the rounding of the constant itself is observable")

    print()
    print("=" * 74)
    print("3. FORWARD PASS")
    print("=" * 74)
    seq_len = 3
    attention_mask = torch.zeros(1, 1, seq_len, seq_len)  # additive mask, 0 = attend
    position_ids = torch.arange(seq_len).unsqueeze(0) + 1
    with torch.no_grad():
        out = model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=embeds,
            kv_cache=None,
        )
    print(f"  inputs_embeds {list(embeds.shape)} -> logits {list(out['logits'].shape)}")
    print(f"  logits dtype: {out['logits'].dtype}  <- forced to float32 for a stable softmax")
    print(f"  keys returned: {list(out.keys())}   (no kv_cache: we passed None)")

    with torch.no_grad():
        out = model(
            attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=embeds, kv_cache=KVCache()
        )
    print(f"  keys with a cache: {list(out.keys())}   <- inference.py feeds this back in")

    print()
    print("=" * 74)
    print("4. THE TIED HEAD IS A DOT PRODUCT WITH THE VOCABULARY")
    print("=" * 74)
    hidden = torch.randn(1, 1, config.hidden_size)
    with torch.no_grad():
        logits = model.lm_head(hidden)
        manual = hidden @ model.get_input_embeddings().weight.t()
    print(f"  lm_head(h) vs h @ embed_tokens.weight^T : max diff {(logits - manual).abs().max():.3e}")
    print("  Because the weights are tied, 'the score of token t' literally is 'how")
    print("  aligned is my hidden state with the embedding of token t'.")

    print()
    print("=" * 74)
    print("5. PARAMETER BUDGET OF THE REAL TEXT MODEL")
    print("=" * 74)
    real = GemmaConfig(
        vocab_size=257152,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
    )
    embedding = real.vocab_size * real.hidden_size
    per_layer_mlp = 3 * real.hidden_size * real.intermediate_size
    per_layer_attn = (
        real.hidden_size * real.num_attention_heads * real.head_dim
        + 2 * real.hidden_size * real.num_key_value_heads * real.head_dim
        + real.num_attention_heads * real.head_dim * real.hidden_size
    )
    total = embedding + real.num_hidden_layers * (per_layer_mlp + per_layer_attn)
    print(f"  embedding table (shared with the head) : {embedding:>13,}")
    print(f"  MLP  per layer (gate + up + down)      : {per_layer_mlp:>13,}")
    print(f"  attention per layer (q,k,v,o)          : {per_layer_attn:>13,}")
    print(f"  x {real.num_hidden_layers} layers + embedding                : {total:>13,}  (~{total / 1e9:.2f}B)")
    vision = 412_442_352
    print(f"  + the vision tower of Episode 10       : {vision:>13,}")
    print(f"  = the '3B' in paligemma-3b             : {total + vision:>13,}  (~{(total + vision) / 1e9:.2f}B)")
    print(f"\n  Note the embedding table alone is {100 * embedding / total:.0f}% of the text model: that is what")
    print("  a 257k-token vocabulary costs, and why tying it (Episode 12) matters.")
    print(f"  Note also the MLP is {per_layer_mlp / per_layer_attn:.0f}x bigger than attention in every layer.")


if __name__ == "__main__":
    main()
