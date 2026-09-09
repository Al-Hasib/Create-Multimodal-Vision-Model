"""Episode 12 -- the config objects and weight tying.

Run:  python check_config_and_tying.py
"""

import torch

from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration

# This is the shape of paligemma-3b-pt-224/config.json, with the real values.
REAL_CONFIG = {
    "vision_config": {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "num_image_tokens": 256,
    },
    "text_config": {
        "vocab_size": 257152,
        "hidden_size": 2048,
        "intermediate_size": 16384,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
    },
    "image_token_index": 256000,
    "projection_dim": 2048,
    "pad_token_id": 0,
}

# The same structure, shrunk so we can instantiate it instantly.
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
        "vocab_size": 1000,
        "hidden_size": 48,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 12,
    },
    "image_token_index": 900,
    "projection_dim": 48,
    "pad_token_id": 0,
}


def main() -> None:
    print("=" * 74)
    print("1. ONE JSON FILE, TWO CONFIG OBJECTS")
    print("=" * 74)
    config = PaliGemmaConfig(**REAL_CONFIG)
    print(f"  PaliGemmaConfig.vision_config -> SiglipVisionConfig (hidden {config.vision_config.hidden_size})")
    print(f"  PaliGemmaConfig.text_config   -> GemmaConfig        (hidden {config.text_config.hidden_size})")
    print("\n  Two things are derived in __init__ rather than read from the file:")
    print(f"    text_config.num_image_tokens = (image_size / patch_size)^2 = {config.text_config.num_image_tokens}")
    print(f"    vision_config.projection_dim = projection_dim            = {config.vision_config.projection_dim}")
    print("  The projector's job is exactly to bridge those two hidden sizes:")
    print(f"    vision {config.vision_config.hidden_size} --Linear--> {config.vision_config.projection_dim}"
          f" == text hidden {config.text_config.hidden_size}")
    print(f"\n  vocab_size is taken from the text config: {config.vocab_size}")
    print(f"  image_token_index = {config.image_token_index}, i.e. <image> is id 256000, just")
    print("  above the 256000-token text vocabulary. The <locNNNN>/<segNNN> tokens fill")
    print("  the rest up to 257152.")

    print()
    print("=" * 74)
    print("2. HEAD COUNTS: A PREVIEW OF EPISODES 18 AND 19")
    print("=" * 74)
    text = config.text_config
    print(f"  num_attention_heads (queries) : {text.num_attention_heads}")
    print(f"  num_key_value_heads           : {text.num_key_value_heads}   <- multi-query attention!")
    print(f"  head_dim                      : {text.head_dim}")
    print(f"  hidden_size                   : {text.hidden_size}")
    print(f"\n  head_dim is a config entry of its own, not hidden_size / num_heads."
          f" Here they")
    print(f"  happen to agree ({text.num_attention_heads} x {text.head_dim} ="
          f" {text.num_attention_heads * text.head_dim} = hidden_size), but our code never assumes it:")
    print(f"    q_proj : {text.hidden_size} -> num_heads    x head_dim ="
          f" {text.num_attention_heads * text.head_dim}")
    print(f"    k_proj : {text.hidden_size} -> num_kv_heads x head_dim ="
          f" {text.num_key_value_heads * text.head_dim}")
    print(f"    v_proj : {text.hidden_size} -> num_kv_heads x head_dim ="
          f" {text.num_key_value_heads * text.head_dim}")
    print(f"    o_proj : {text.num_attention_heads * text.head_dim} -> {text.hidden_size}")
    print(f"  {text.num_attention_heads} query heads share {text.num_key_value_heads} key/value head, so we cache"
          f" {text.num_attention_heads}x fewer keys and")
    print("  values than a plain multi-head model would (Episode 18).")

    print()
    print("=" * 74)
    print("3. WEIGHT TYING")
    print("=" * 74)
    model = PaliGemmaForConditionalGeneration(PaliGemmaConfig(**TINY_CONFIG))
    embed = model.language_model.get_input_embeddings().weight
    head = model.language_model.lm_head.weight
    print(f"  embed_tokens.weight : {list(embed.shape)}  [Vocab_Size, Hidden_Size]")
    print(f"  lm_head.weight      : {list(head.shape)}  [Vocab_Size, Hidden_Size]  <- same shape")
    print(f"\n  before tie_weights(): same object? {embed is head}")
    model.tie_weights()
    embed = model.language_model.get_input_embeddings().weight
    head = model.language_model.lm_head.weight
    print(f"  after  tie_weights(): same object? {embed is head}")

    with torch.no_grad():
        embed[0, 0] = 123.0
    print(f"  writing to embed_tokens changes lm_head too: {head[0, 0].item()}")

    print("\n  Why tie them? The embedding maps a token id to a vector; the LM head maps")
    print("  a vector back to a score per token. They are inverse operations, so sharing")
    print("  one matrix is a sensible inductive bias -- and it is free memory:")
    real = PaliGemmaConfig(**REAL_CONFIG).text_config
    saved = real.vocab_size * real.hidden_size
    print(f"    {real.vocab_size:,} x {real.hidden_size:,} = {saved:,} parameters")
    print(f"    = {saved * 2 / 1e9:.2f} GB in bfloat16 that we do NOT have to store")
    print("\n  This is also why `utils.load_hf_model` calls tie_weights() after")
    print("  load_state_dict: the checkpoint has no lm_head.weight entry at all, which")
    print("  is also why it loads with strict=False.")

    print()
    print("=" * 74)
    print("4. THE SKELETON WE JUST WIRED")
    print("=" * 74)
    for name, module in model.named_children():
        parameters = sum(p.numel() for p in module.parameters())
        print(f"    {name:<24} {type(module).__name__:<35} {parameters:>10,} params")
    print("\n  vision_tower is finished (Episode 10). multi_modal_projector is one")
    print("  Linear. language_model is still a stub -- forward() raises NotImplementedError")
    print("  until Episode 13.")


if __name__ == "__main__":
    main()
