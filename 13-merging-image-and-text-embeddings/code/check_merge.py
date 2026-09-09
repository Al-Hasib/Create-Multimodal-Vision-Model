"""Episode 13 -- watch the image embeddings replace the <image> placeholders.

`_merge_input_ids_with_image_features` is the heart of "multimodal": after this
function there is no image and no text any more, only one sequence of vectors.

Run:  python check_merge.py
"""

import torch

from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration

TINY_CONFIG = {
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_size": 32,
        "patch_size": 16,  # -> 4 image tokens, small enough to print
    },
    "text_config": {
        "vocab_size": 1000,
        "hidden_size": 8,  # tiny hidden size so we can print whole vectors
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
    },
    "image_token_index": 900,
    "projection_dim": 8,
    "pad_token_id": 0,
}

IMAGE, BOS, PAD = 900, 2, 0


def main() -> None:
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=140)

    config = PaliGemmaConfig(**TINY_CONFIG)
    model = PaliGemmaForConditionalGeneration(config).eval()
    model.tie_weights()

    # 4 image placeholders, <bos>, two text tokens, then one pad.
    input_ids = torch.tensor([[IMAGE, IMAGE, IMAGE, IMAGE, BOS, 42, 77, PAD]])
    batch_size, seq_len = input_ids.shape
    num_image_tokens = 4

    print("=" * 74)
    print("1. THE INPUT SEQUENCE")
    print("=" * 74)
    print(f"  input_ids = {input_ids.tolist()[0]}")
    print(f"              {'  '.join(['img'] * num_image_tokens)}  bos  txt txt pad")
    print(f"  image_token_index = {config.image_token_index}, pad_token_id = {model.pad_token_id}")

    with torch.no_grad():
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        image_features = model.multi_modal_projector(model.vision_tower(torch.randn(1, 3, 32, 32)))
        merged, mask, position_ids = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, torch.ones_like(input_ids)
        )

    print()
    print("=" * 74)
    print("2. THE THREE MASKS")
    print("=" * 74)
    text_mask = (input_ids != config.image_token_index) & (input_ids != model.pad_token_id)
    print(f"  text_mask  (real text)  : {text_mask.int().tolist()[0]}")
    print(f"  image_mask (placeholder): {(input_ids == config.image_token_index).int().tolist()[0]}")
    print(f"  pad_mask   (padding)    : {(input_ids == model.pad_token_id).int().tolist()[0]}")
    print("\n  They partition the sequence: exactly one is True at every position.")
    print("  Note the ORDER of the three writes matters -- pad is zeroed last, so a")
    print("  pad position cannot keep whatever the embedding table returned for id 0.")

    print()
    print("=" * 74)
    print("3. WHERE EACH VECTOR ENDED UP")
    print("=" * 74)
    print(f"  image_features {list(image_features.shape)}  [Batch_Size, Num_Patches, Projection_Dim]")
    print(f"  inputs_embeds  {list(inputs_embeds.shape)}  [Batch_Size, Seq_Len, Hidden_Size]")
    print(f"  merged         {list(merged.shape)}  <- same shape as inputs_embeds\n")

    scaled = image_features / (config.hidden_size**0.5)
    for position in range(seq_len):
        if position < num_image_tokens:
            source = "image" if torch.allclose(merged[0, position], scaled[0, position]) else "???"
        elif input_ids[0, position] == PAD:
            source = "zeros" if merged[0, position].abs().sum() == 0 else "???"
        else:
            source = "text" if torch.allclose(merged[0, position], inputs_embeds[0, position]) else "???"
        print(f"    position {position} (id {input_ids[0, position].item():>3}) <- {source:<5}  {merged[0, position][:4]}")

    print()
    print("=" * 74)
    print("4. WHY masked_scatter AND NOT torch.where")
    print("=" * 74)
    print("  torch.where(cond, a, b) needs a, b and cond broadcastable to one shape.")
    print(f"  But image_features is [1, {num_image_tokens}, 8] while the sequence is [1, {seq_len}, 8]:")
    print("  the 4 image vectors have to be *spread* over the 4 True positions of the")
    print("  mask. masked_scatter walks the mask and consumes the source tensor in")
    print("  order, which is exactly that operation.")
    print("\n  It also silently requires that the number of True entries equals the")
    print("  number of source vectors -- the reason num_image_tokens in the processor")
    print("  and (image_size / patch_size)^2 in the vision tower must agree.")

    print()
    print("=" * 74)
    print("5. THE MYSTERIOUS DIVISION BY sqrt(hidden_size)")
    print("=" * 74)
    normalizer = config.hidden_size**0.5
    print(f"  scaled_image_features = image_features / sqrt({config.hidden_size}) = / {normalizer:.3f}")
    print(f"  ||image_features[0,0]|| = {image_features[0, 0].norm():.4f}")
    print(f"  ||merged[0,0]||         = {merged[0, 0].norm():.4f}")
    print("\n  Why? Because GemmaModel.forward multiplies the *whole* embedding sequence")
    print("  by sqrt(hidden_size) on its way in (Episode 16) -- Gemma's way of scaling")
    print("  its token embeddings. Text embeddings are trained expecting that scale-up;")
    print("  the projected image features are not, so we pre-divide to cancel it.")
    print("  Take this line out and the image tokens arrive ~45x too loud.")

    print()
    print("=" * 74)
    print("6. STILL MISSING (Episode 15)")
    print("=" * 74)
    print(f"  attention mask returned : {mask}")
    print(f"  position ids returned   : {position_ids}")
    print("  The embeddings are merged, but nobody has told the model which positions")
    print("  these tokens occupy or who is allowed to attend to whom.")


if __name__ == "__main__":
    main()
