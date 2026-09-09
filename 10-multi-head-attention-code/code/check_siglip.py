"""Episode 10 -- the vision tower is finished. Let's verify it.

Three checks:
  1. our SiglipAttention agrees with PyTorch's fused kernel, weight for weight
  2. information now flows between patches (the Episode 08 check, re-run)
  3. the real So400m tower has the parameter count we expect

Run:  python check_siglip.py
"""

import torch
import torch.nn.functional as F

from modeling_siglip import SiglipAttention, SiglipVisionConfig, SiglipVisionModel

torch.manual_seed(0)


def check_against_pytorch() -> None:
    print("=" * 74)
    print("1. OUR ATTENTION == F.scaled_dot_product_attention")
    print("=" * 74)
    config = SiglipVisionConfig(hidden_size=64, num_attention_heads=8)
    attention = SiglipAttention(config).eval()
    hidden_states = torch.randn(2, 10, config.hidden_size)

    with torch.no_grad():
        ours, weights = attention(hidden_states)

        # The same maths with PyTorch's kernel, reusing our projections.
        batch_size, seq_len, _ = hidden_states.shape
        shape = (batch_size, seq_len, attention.num_heads, attention.head_dim)
        q = attention.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = attention.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = attention.v_proj(hidden_states).view(shape).transpose(1, 2)
        reference = F.scaled_dot_product_attention(q, k, v)
        reference = reference.transpose(1, 2).reshape(batch_size, seq_len, config.hidden_size)
        reference = attention.out_proj(reference)

    print(f"  attn_weights shape  {list(weights.shape)}   [Batch_Size, Num_Heads, Num_Patches, Num_Patches]")
    print(f"  rows sum to 1       {torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))}")
    print(f"  max |ours - torch|  {(ours - reference).abs().max():.3e}")
    print("  Same numbers. PyTorch's version never materialises the score matrix,")
    print("  which is the only reason to prefer it in production.")


def check_information_flow() -> None:
    print()
    print("=" * 74)
    print("2. PATCHES NOW TALK TO EACH OTHER")
    print("=" * 74)
    config = SiglipVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8
    )
    model = SiglipVisionModel(config).eval()
    num_patches = (config.image_size // config.patch_size) ** 2

    pixel_values = torch.randn(1, 3, config.image_size, config.image_size)
    perturbed = pixel_values.clone()
    perturbed[0, :, : config.patch_size, : config.patch_size] += 5.0  # patch 0 only

    with torch.no_grad():
        delta = (model(pixel_values) - model(perturbed))[0].abs().max(dim=-1).values

    print("  Change patch 0, measure how far each output token moved:")
    for index in (0, 1, 2, num_patches - 1):
        print(f"    token {index:<3}: {delta[index]:.4f}")
    print(f"\n  smallest movement over all {num_patches} tokens = {delta.min():.4f}")
    print("  In Episode 08 every token except token 0 moved by exactly 0. Attention is")
    print("  the only component that mixes positions -- everything else is per token.")


def check_real_tower() -> None:
    print()
    print("=" * 74)
    print("3. THE TOWER PaliGemma ACTUALLY LOADS")
    print("=" * 74)
    config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        num_image_tokens=256,
    )
    model = SiglipVisionModel(config)
    total = sum(p.numel() for p in model.parameters())
    per_layer = sum(p.numel() for p in model.vision_model.encoder.layers[0].parameters())

    print(f"  hidden_size {config.hidden_size}, {config.num_hidden_layers} layers, {config.num_attention_heads} heads"
          f" -> head_dim {config.hidden_size // config.num_attention_heads}")
    print(f"  parameters per encoder layer : {per_layer:>12,}")
    print(f"  parameters in the whole tower: {total:>12,}  (~{total / 1e6:.0f}M, the 'So400m' family)")
    print(f"  image tokens produced        : {(config.image_size // config.patch_size) ** 2}")

    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    print(f"\n  [1, 3, 224, 224] -> {list(out.shape)}")
    print("  256 vectors of 1152 dims. Episode 12 projects them to Gemma's 2048.")


def main() -> None:
    check_against_pytorch()
    check_information_flow()
    check_real_tower()

    print()
    print("=" * 74)
    print("4. EXERCISES")
    print("=" * 74)
    print("  a) Delete the `* self.scale` and print the max attention weight. How close")
    print("     to a one-hot distribution does the softmax get?")
    print("  b) Set num_attention_heads=1152 (head_dim 1). What still works, what does")
    print("     the model lose?")
    print("  c) Remove out_proj (return the concatenated heads directly). Explain why")
    print("     the heads can then never combine their findings.")
    print("  d) Feed the same image twice in a batch and confirm the two outputs match:")
    print("     attention must never leak across the batch dimension.")


if __name__ == "__main__":
    main()
