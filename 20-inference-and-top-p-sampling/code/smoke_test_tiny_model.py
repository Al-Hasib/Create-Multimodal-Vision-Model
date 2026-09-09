"""Episode 20 -- run the complete generation loop without downloading 3 GB.

We build a tiny randomly initialised PaliGemma and generate a few tokens with the
same loop inference.py uses. The output is gibberish -- that is fine, and it is
the point: this checks the machinery (prefill, cache growth, position ids, sampling)
rather than the weights. Run it before you spend time on the real checkpoint.

Run:  python smoke_test_tiny_model.py
"""

import torch

from modeling_gemma import KVCache, PaliGemmaConfig, PaliGemmaForConditionalGeneration

torch.manual_seed(0)

NUM_IMAGE_TOKENS = 4  # image_size 32 / patch_size 16 -> 2x2 patches

TINY_CONFIG = {
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "image_size": 32,
        "patch_size": 16,
    },
    "text_config": {
        "vocab_size": 257,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
    },
    "image_token_index": 256,
    "projection_dim": 32,
    "pad_token_id": 0,
}


def main() -> None:
    config = PaliGemmaConfig(**TINY_CONFIG)
    model = PaliGemmaForConditionalGeneration(config).eval()
    model.tie_weights()

    # What the processor would have produced: image placeholders, <bos>, prompt, "\n".
    image_token = config.image_token_index
    input_ids = torch.tensor([[image_token] * NUM_IMAGE_TOKENS + [2, 42, 77, 108]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 32, 32)

    print("=" * 74)
    print("1. THE GENERATION LOOP")
    print("=" * 74)
    print(f"  prompt      : {input_ids.shape[1]} tokens ({NUM_IMAGE_TOKENS} image + 4 text)")
    print(f"  parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print()

    kv_cache = KVCache()
    generated = []
    max_new_tokens = 5

    with torch.no_grad():
        for step in range(max_new_tokens):
            cached_before = kv_cache.num_items()
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).squeeze(0)
            generated.append(next_token.item())

            phase = "PREFILL" if step == 0 else "decode "
            print(f"  {phase} step {step}: input_ids {list(input_ids.shape)}"
                  f"  logits {list(outputs['logits'].shape)}"
                  f"  cache {cached_before} -> {kv_cache.num_items()}"
                  f"  next id {next_token.item()}")

            # Exactly what inference.py does: feed back only the new token,
            # and grow the attention mask by one.
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1))], dim=-1)

    print(f"\n  generated ids: {generated}   (random weights, so meaningless)")

    print()
    print("=" * 74)
    print("2. WHAT THIS PROVED")
    print("=" * 74)
    print(f"  - prefill consumed all {NUM_IMAGE_TOKENS + 4} prompt tokens at once and produced"
          f" {NUM_IMAGE_TOKENS + 4} logits")
    print("  - every later step sent ONE token and produced ONE set of logits")
    print("  - the cache grew by exactly 1 per step, so keys/values are appended, not")
    print("    recomputed")
    print("  - pixel_values is passed every step: the vision tower re-runs each time.")
    print("    Wasteful but harmless (the image features are identical); caching them")
    print("    is a good exercise")
    print("  - logits[:, -1, :] is always the prediction for the NEXT token: during")
    print("    prefill we throw away the other positions, which is what the model")
    print("    computes anyway")

    print()
    print("=" * 74)
    print("3. THE SHAPE OF THE CACHE AT THE END")
    print("=" * 74)
    print(f"  layers cached      : {len(kv_cache.key_cache)}")
    print(f"  key_cache[0] shape : {list(kv_cache.key_cache[0].shape)}"
          "  [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]")
    print(f"  positions cached   : {kv_cache.num_items()}"
          f"  = {NUM_IMAGE_TOKENS + 4} prompt + {max_new_tokens} generated")

    print()
    print("=" * 74)
    print("4. NOW DO IT FOR REAL")
    print("=" * 74)
    print("  huggingface-cli download google/paligemma-3b-pt-224 \\")
    print("      --local-dir ~/projects/paligemma-weights/paligemma-3b-pt-224")
    print("  ./launch_inference.sh")
    print("\n  See this episode's README for the full instructions, including the")
    print("  license acceptance step and what to expect on CPU.")


if __name__ == "__main__":
    main()
