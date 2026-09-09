"""Episode 06 -- exercise the code we just wrote: config + vision embeddings.

Run:  python check_embeddings.py
"""

import torch

from modeling_siglip import SiglipVisionConfig, SiglipVisionEmbeddings


def describe(config: SiglipVisionConfig, name: str) -> None:
    embeddings = SiglipVisionEmbeddings(config)
    pixel_values = torch.randn(2, config.num_channels, config.image_size, config.image_size)
    out = embeddings(pixel_values)

    num_parameters = sum(p.numel() for p in embeddings.parameters())
    print(f"\n{name}")
    print(f"  image_size {config.image_size}, patch_size {config.patch_size}, hidden_size {config.hidden_size}")
    print(f"  num_patches                     = {embeddings.num_patches}")
    print(f"  patch_embedding.weight          = {list(embeddings.patch_embedding.weight.shape)}"
          f"  [Embed_Dim, Channels, Patch, Patch]")
    print(f"  position_embedding.weight       = {list(embeddings.position_embedding.weight.shape)}"
          f"  [Num_Patches, Embed_Dim]")
    print(f"  pixel_values {list(pixel_values.shape)} -> embeddings {list(out.shape)}")
    print(f"  parameters                      = {num_parameters:,}")


def main() -> None:
    print("=" * 74)
    print("1. THE DEFAULTS (ViT-Base numbers) AND THE REAL PaliGemma TOWER")
    print("=" * 74)
    describe(SiglipVisionConfig(), "SiglipVisionConfig()  -- the defaults in our file")
    describe(
        # These come from the vision_config block of paligemma-3b-pt-224/config.json.
        SiglipVisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
            num_image_tokens=256,
        ),
        "SigLIP-So400m         -- what paligemma-3b-pt-224 actually loads",
    )

    print()
    print("=" * 74)
    print("2. THE POSITION IDS BUFFER")
    print("=" * 74)
    config = SiglipVisionConfig()
    embeddings = SiglipVisionEmbeddings(config)
    print(f"  position_ids       = {embeddings.position_ids[0, :8].tolist()} ... "
          f"{embeddings.position_ids[0, -2:].tolist()}   shape {list(embeddings.position_ids.shape)}")
    print(f"  in state_dict()    = {'position_ids' in embeddings.state_dict()}"
          "   <- persistent=False, so it is never saved or loaded")
    print("  It is registered as a buffer (not a parameter) so `.to(device)` moves it")
    print("  with the module, but the optimizer never sees it.")

    print()
    print("=" * 74)
    print("3. EVERY PATCH GETS ITS OWN POSITION VECTOR")
    print("=" * 74)
    with torch.no_grad():
        # Feed a constant image: all patches produce the identical patch embedding,
        # so any difference between tokens has to come from the position embedding.
        constant = torch.ones(1, 3, config.image_size, config.image_size)
        out = embeddings(constant)
    print(f"  constant image -> tokens {list(out.shape)}")
    print(f"  token 0 == token 1 ? {torch.allclose(out[0, 0], out[0, 1])}")
    print(f"  ||token0 - token1|| = {(out[0, 0] - out[0, 1]).norm():.4f}   <- purely positional")

    print()
    print("=" * 74)
    print("4. EXERCISES")
    print("=" * 74)
    print("  a) Set patch_size=32 and predict num_patches before running it.")
    print("  b) Why is `padding='valid'` important? Try padding='same' with an image")
    print("     size that is not a multiple of the patch size and watch num_patches")
    print("     disagree with the actual output length.")
    print("  c) The position embedding table has a fixed size. What breaks if you feed")
    print("     a 448x448 image to a tower trained at 224x224?")


if __name__ == "__main__":
    main()
