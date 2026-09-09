"""Episode 05 -- how an image becomes a sequence of tokens.

The one idea of the Vision Transformer: cut the image into fixed-size patches, flatten
each patch, project it to `hidden_size`, and pretend the result is a sentence. This
script shows that the strided convolution we use in `SiglipVisionEmbeddings` is
exactly "cut into patches, then apply one linear layer per patch".

Run:  python vit_patch_embedding.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

BATCH_SIZE, CHANNELS, IMAGE_SIZE, PATCH_SIZE, EMBED_DIM = 2, 3, 224, 16, 768


def main() -> None:
    num_patches_per_side = IMAGE_SIZE // PATCH_SIZE
    num_patches = num_patches_per_side**2

    print("=" * 74)
    print("1. THE SHAPES")
    print("=" * 74)
    print(f"  image        : [{BATCH_SIZE}, {CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}]")
    print(f"  patch grid   : {num_patches_per_side} x {num_patches_per_side} = {num_patches} patches")
    print(f"  one patch    : {CHANNELS} x {PATCH_SIZE} x {PATCH_SIZE} = {CHANNELS * PATCH_SIZE * PATCH_SIZE} numbers")
    print(f"  token        : {EMBED_DIM} numbers after the projection")
    print(f"  sequence     : [{BATCH_SIZE}, {num_patches}, {EMBED_DIM}]  <- this is what the transformer sees")

    images = torch.randn(BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    print()
    print("=" * 74)
    print("2. A STRIDED CONV *IS* PATCH-AND-PROJECT")
    print("=" * 74)
    conv = nn.Conv2d(CHANNELS, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, padding="valid")

    # (a) the way we write it in modeling_siglip.py
    conv_tokens = conv(images)  # [B, Embed_Dim, H/P, W/P]
    print(f"  conv(images)                    -> {list(conv_tokens.shape)}")
    conv_tokens = conv_tokens.flatten(2).transpose(1, 2)
    print(f"  .flatten(2).transpose(1, 2)     -> {list(conv_tokens.shape)}")

    # (b) the literal ViT-paper description, using the same weights
    patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(BATCH_SIZE, num_patches, -1)
    weight = conv.weight.reshape(EMBED_DIM, -1)  # one row per output dimension
    manual_tokens = patches @ weight.t() + conv.bias
    print(f"  unfold + reshape + linear       -> {list(manual_tokens.shape)}")
    print(f"\n  max |conv - manual| = {(conv_tokens - manual_tokens).abs().max():.3e}  -> the same operation")
    print("  Stride == kernel size is what makes the patches non-overlapping, and")
    print("  padding='valid' means no pixel is invented at the border.")

    print()
    print("=" * 74)
    print("3. THE TRANSFORMER HAS NO IDEA WHERE A PATCH CAME FROM")
    print("=" * 74)
    tokens = conv_tokens
    shuffled = tokens[:, torch.randperm(num_patches), :]
    print("  Self-attention is permutation equivariant: shuffle the tokens and you get")
    print("  the same set of outputs, just reordered. A patch at the top-left and a")
    print("  patch at the bottom-right are indistinguishable to it.")

    position_embedding = nn.Embedding(num_patches, EMBED_DIM)
    position_ids = torch.arange(num_patches).expand((1, -1))
    print(f"\n  position_ids                    -> {list(position_ids.shape)}  (0, 1, 2, ..., {num_patches - 1})")
    print(f"  position_embedding(position_ids) -> {list(position_embedding(position_ids).shape)}")
    with_positions = tokens + position_embedding(position_ids)
    print(f"  tokens + positions              -> {list(with_positions.shape)}  (broadcast over the batch)")
    print("\n  SigLIP uses *learned absolute* position embeddings -- one trainable vector")
    print("  per slot, simply added to the patch embedding. Compare that with the text")
    print("  side of the model, which rotates Q and K instead (RoPE, Episode 19).")
    print(f"  Different vectors per slot: {not torch.allclose(position_embedding(position_ids)[0, 0], position_embedding(position_ids)[0, 1])}")

    print()
    print("=" * 74)
    print("4. NO CAUSAL MASK HERE")
    print("=" * 74)
    print("  A language model must not look at future tokens. An image is not causal:")
    print("  every patch may attend to every other patch, which is why")
    print("  `SiglipAttention.forward` takes no attention_mask argument at all.")
    print("  Each of the 196 output vectors is a *contextualized* patch embedding.")

    print()
    print("=" * 74)
    print("5. THE CONFIGURATION WE ARE ABOUT TO CODE (Episode 06)")
    print("=" * 74)
    print("  paligemma-3b-pt-224 uses the SigLIP-So400m vision tower:")
    print("    image_size 224, patch_size 14 -> 256 image tokens")
    print("    hidden_size 1152, intermediate_size 4304, 27 layers, 16 heads")
    print("  The defaults in SiglipVisionConfig are the smaller ViT-Base numbers")
    print("  (768 / 3072 / 12 / 12, patch 16); the real values come from config.json.")


if __name__ == "__main__":
    main()
