"""Episode 08 -- the encoder stack runs end to end (with attention still stubbed).

`SiglipAttention` is a placeholder in this episode's snapshot: it returns its input
unchanged. That is deliberate -- it lets us verify the plumbing (shapes, residuals,
LayerNorm placement, the MLP) *and* it makes the role of attention impossible to
miss: without it, token i never learns anything about token j.

Run:  python check_encoder.py
"""

import torch

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


def main() -> None:
    config = SiglipVisionConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8
    )
    model = SiglipVisionModel(config).eval()

    print("=" * 74)
    print("1. SHAPES THROUGH THE WHOLE TOWER")
    print("=" * 74)
    pixel_values = torch.randn(2, 3, config.image_size, config.image_size)
    with torch.no_grad():
        out = model(pixel_values)
    num_patches = (config.image_size // config.patch_size) ** 2
    print(f"  pixel_values  {list(pixel_values.shape)}")
    print(f"  -> embeddings [{2}, {num_patches}, {config.hidden_size}]")
    print(f"  -> encoder    ({config.num_hidden_layers} identical layers, shape preserving)")
    print(f"  -> post_layernorm")
    print(f"  -> output     {list(out.shape)}   [Batch_Size, Num_Patches, Embed_Dim]")
    print("\n  Notice the tower has no pooling head and no [CLS] token: PaliGemma wants")
    print("  all the patch embeddings, one language token per patch.")

    print()
    print("=" * 74)
    print("2. THE LAYER IS PRE-NORM")
    print("=" * 74)
    print("  residual = x;  x = layer_norm1(x);  x = attn(x);   x = residual + x")
    print("  residual = x;  x = layer_norm2(x);  x = mlp(x);    x = residual + x")
    print("\n  The normalization sits INSIDE the branch, never on the residual highway,")
    print("  so gradients reach layer 0 unscaled. Post-norm (`layer_norm(residual + x)`)")
    print("  is the original 2017 recipe and needs warmup to train at this depth.")
    layer = model.vision_model.encoder.layers[0]
    print(f"\n  layer 0 submodules: {[name for name, _ in layer.named_children()]}")

    print()
    print("=" * 74)
    print("3. THE FFN EXPANDS BY 4x AND HOLDS MOST OF THE PARAMETERS")
    print("=" * 74)
    mlp_parameters = sum(p.numel() for p in layer.mlp.parameters())
    # Attention is still a stub with zero parameters, so count what it *will* have:
    # four square projections (q, k, v, out), each with a bias.
    d = config.hidden_size
    attention_parameters = 4 * (d * d + d)
    print(f"  fc1: {config.hidden_size} -> {config.intermediate_size}   (gelu, tanh approximation)")
    print(f"  fc2: {config.intermediate_size} -> {config.hidden_size}")
    print(f"  mlp parameters                  {mlp_parameters:>9,}")
    print(f"  attention parameters (Episode 10) {attention_parameters:>7,}  <- 4 x (d*d + d), zero for now")
    print(f"  the FFN will hold {100 * mlp_parameters / (mlp_parameters + attention_parameters):.0f}% of the layer, "
          f"because intermediate_size = 4 x hidden_size")
    print("  Attention mixes information *between* tokens; the FFN is applied to each")
    print("  token independently and is where most of the knowledge is stored.")

    print()
    print("=" * 74)
    print("4. PROOF THAT WE ARE STILL MISSING ATTENTION")
    print("=" * 74)
    perturbed = pixel_values.clone()
    perturbed[0, :, : config.patch_size, : config.patch_size] += 5.0  # only patch 0
    with torch.no_grad():
        out2 = model(perturbed)
    delta = (out - out2)[0].abs().max(dim=-1).values
    print("  We change patch 0 of sample 0 only, then measure how much each output")
    print("  token moved:")
    print(f"    token 0 : {delta[0]:.4f}")
    print(f"    token 1 : {delta[1]:.4f}")
    print(f"    tokens 1..{num_patches - 1} max: {delta[1:].max():.4f}")
    if delta[1:].max() < 1e-6:
        print("\n  Every other token is bit-identical: with a stubbed attention this is")
        print("  just an MLP applied per patch. Re-run this exact check in Episode 10")
        print("  and all tokens will move -- that is what attention buys us.")

    print()
    print("=" * 74)
    print("5. EXERCISES")
    print("=" * 74)
    print("  a) Move layer_norm1 outside the branch (post-norm) and print the gradient")
    print("     norm at layer 0 for both variants with num_hidden_layers=27.")
    print("  b) Replace gelu with relu. How large is the numerical difference at the")
    print("     output? (Then check what the real SigLIP checkpoint was trained with.)")
    print("  c) Why does `SiglipEncoder.forward` take no attention mask at all?")


if __name__ == "__main__":
    main()
