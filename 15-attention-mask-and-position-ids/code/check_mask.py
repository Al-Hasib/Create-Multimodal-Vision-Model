"""Episode 15 -- the attention mask and the position ids, in both phases.

PaliGemma's mask is the most surprising part of the whole model: it is NOT causal
over the prompt. This script prints what our code actually builds.

Run:  python check_mask.py
"""

import torch

from modeling_gemma import KVCache, PaliGemmaConfig, PaliGemmaForConditionalGeneration

TINY_CONFIG = {
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "image_size": 32,
        "patch_size": 16,  # -> 4 image tokens
    },
    "text_config": {
        "vocab_size": 1000,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
    },
    "image_token_index": 900,
    "projection_dim": 16,
    "pad_token_id": 0,
}

IMAGE, BOS, NEWLINE = 900, 2, 108


def main() -> None:
    torch.manual_seed(0)
    config = PaliGemmaConfig(**TINY_CONFIG)
    model = PaliGemmaForConditionalGeneration(config).eval()
    model.tie_weights()

    # 4 image tokens, <bos>, two prompt tokens, newline: the processor's output.
    input_ids = torch.tensor([[IMAGE, IMAGE, IMAGE, IMAGE, BOS, 42, 77, NEWLINE]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 32, 32)

    print("=" * 74)
    print("1. PREFILL: kv_cache IS EMPTY")
    print("=" * 74)
    kv_cache = KVCache()
    with torch.no_grad():
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        image_features = model.multi_modal_projector(model.vision_tower(pixel_values))
        _, mask, position_ids = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )
    print(f"  kv_cache.num_items() = {kv_cache.num_items()}")
    print(f"  mask shape   {list(mask.shape)}  [Batch_Size, Num_Heads (broadcast), Q_Len, KV_Len]")
    print(f"  unique values in the mask: {mask.unique().tolist()}   <- all zeros, nothing is blocked")
    print(f"  position_ids {position_ids.tolist()[0]}")
    print("\n  A CAUSAL mask would look like this instead:")
    q_len = input_ids.shape[1]
    causal = torch.full((q_len, q_len), float("-inf")).triu(1)
    for row in causal[:4]:
        print("    " + " ".join("  . " if v == 0 else " -inf" for v in row))

    print()
    print("=" * 74)
    print("2. WHY NO CAUSAL MASK OVER THE PROMPT?")
    print("=" * 74)
    print("  PaliGemma is a PREFIX-LM. The prompt (image tokens + text prompt) is the")
    print("  prefix, and inside it attention is bidirectional:")
    print("    - image tokens must see each other (an image is not a left-to-right")
    print("      sequence -- exactly the argument from Episode 05)")
    print("    - the prompt is given, never predicted, so letting prompt token 5 look")
    print("      at prompt token 7 leaks nothing")
    print("  Only the *generated* suffix is causal, and that comes for free: during")
    print("  decoding each new token is appended after everything already in the cache,")
    print("  so it can only look backwards.")
    print("\n  The comment in the code says 'This only works when we have no padding'.")
    print("  That is the real restriction: with an all-zero mask, a padded position")
    print("  would be attended to like any other token. Hence the assert in forward():")
    print("      assert torch.all(attention_mask == 1), 'The input cannot be padded'")

    print()
    print("=" * 74)
    print("3. DECODE: ONE QUERY AGAINST A FULL CACHE")
    print("=" * 74)
    # Pretend the prefill already filled the cache for both layers.
    text = config.text_config
    for _ in range(text.num_hidden_layers):
        kv_cache.key_cache.append(torch.zeros(1, text.num_key_value_heads, q_len, text.head_dim))
        kv_cache.value_cache.append(torch.zeros(1, text.num_key_value_heads, q_len, text.head_dim))

    next_ids = torch.tensor([[314]])  # the token we just generated
    next_attention_mask = torch.ones(1, q_len + 1, dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = model.language_model.get_input_embeddings()(next_ids)
        _, mask, position_ids = model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, next_ids, next_attention_mask, kv_cache
        )
    print(f"  kv_cache.num_items() = {kv_cache.num_items()}")
    print(f"  mask shape   {list(mask.shape)}  -> Q_Len 1, KV_Len {kv_cache.num_items() + 1}")
    print(f"  unique values: {mask.unique().tolist()}  <- again all zeros: the new token")
    print("                 is allowed to attend to the whole past, and there is no")
    print("                 future to hide")
    print(f"  position_ids {position_ids.tolist()}  <- a single position, the end of the sequence")

    print()
    print("=" * 74)
    print("4. HOW position_ids IS COMPUTED, AND WHY cumsum")
    print("=" * 74)
    print("  prefill: position_ids = attention_mask.cumsum(-1), with masked slots forced to 1")
    example = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    print(f"    attention_mask {example.tolist()[0]}")
    print(f"    cumsum         {example.cumsum(-1).tolist()[0]}   <- 1-based, not 0-based")
    print("  decode : position_ids = attention_mask.cumsum(-1)[:, -1], the last position only")
    print(f"    attention_mask {next_attention_mask.tolist()[0]}")
    print(f"    cumsum[:, -1]  {next_attention_mask.cumsum(-1)[:, -1].tolist()}")
    print("\n  cumsum (rather than arange) is what makes this padding aware: a padded")
    print("  slot contributes 0 and therefore does not advance the position counter.")
    print("  These indices feed RoPE in Episode 19 -- they are the ONLY way the text")
    print("  model learns about order.")

    print()
    print("=" * 74)
    print("5. THE FULL FORWARD RUNS NOW (with a pass-through language model)")
    print("=" * 74)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
    print(f"  logits {list(outputs['logits'].shape)}  [Batch_Size, Seq_Len, Vocab_Size]")
    print("  The pipeline is complete end to end: pixels -> patches -> projection ->")
    print("  merged embeddings -> mask + positions -> language model -> logits.")
    print("  GemmaModel is still a stub, so those logits are meaningless. Episodes 16")
    print("  to 19 fill it in.")

    print()
    print("=" * 74)
    print("6. EXERCISES")
    print("=" * 74)
    print("  a) Remove the `assert q_len == 1` in the decode branch and pass two new")
    print("     tokens. What exactly goes wrong with the mask that gets built?")
    print("  b) Change the prefill mask to a proper causal mask and (after Episode 20)")
    print("     compare the generated caption. Which is closer to the HF output?")
    print("  c) Feed a batch of two prompts with padding and find every line in")
    print("     _merge_input_ids_with_image_features that would need to change.")


if __name__ == "__main__":
    main()
