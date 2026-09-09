"""Episode 17 -- the decoder layer and the GeGLU feed-forward network.

Run:  python check_decoder_layer.py
"""

import torch
import torch.nn as nn

from modeling_gemma import GemmaConfig, GemmaDecoderLayer, GemmaMLP

torch.manual_seed(0)

CONFIG = dict(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=16,
)


def main() -> None:
    config = GemmaConfig(**CONFIG, pad_token_id=0)
    layer = GemmaDecoderLayer(config, layer_idx=0).eval()

    print("=" * 74)
    print("1. THE LAYER, LINE BY LINE")
    print("=" * 74)
    print("    residual = hidden_states")
    print("    hidden_states = self.input_layernorm(hidden_states)          # RMSNorm")
    print("    hidden_states = self.self_attn(...)                          # mixes tokens")
    print("    hidden_states = residual + hidden_states")
    print("    residual = hidden_states")
    print("    hidden_states = self.post_attention_layernorm(hidden_states) # RMSNorm")
    print("    hidden_states = self.mlp(hidden_states)                      # per token")
    print("    hidden_states = residual + hidden_states")
    print("\n  Structurally identical to the SigLIP encoder layer of Episode 08:")
    print("  pre-norm, two residual branches. The differences are all in the parts:")
    print("    LayerNorm -> RMSNorm, GELU MLP -> GeGLU MLP, and the attention is causal")
    print("    with RoPE and grouped KV heads.")
    print("\n  Note `post_attention_layernorm` normalizes the input to the MLP; despite")
    print("  the name it does not sit after the attention residual add.")

    print()
    print("=" * 74)
    print("2. SHAPES ARE PRESERVED -- THAT IS WHY YOU CAN STACK 18 OF THEM")
    print("=" * 74)
    hidden_states = torch.randn(1, 5, config.hidden_size)
    attention_mask = torch.zeros(1, 1, 5, 5)
    position_ids = torch.arange(5).unsqueeze(0) + 1
    with torch.no_grad():
        out = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=None)
    print(f"  in  {list(hidden_states.shape)} -> out {list(out.shape)}")
    print("  (self_attn is still the Episode 18 stub, so this only checks the wiring.)")

    print()
    print("=" * 74)
    print("3. GeGLU: THREE MATRICES INSTEAD OF TWO")
    print("=" * 74)
    mlp = GemmaMLP(config).eval()
    x = torch.randn(1, 2, config.hidden_size)
    with torch.no_grad():
        gate = nn.functional.gelu(mlp.gate_proj(x), approximate="tanh")
        up = mlp.up_proj(x)
        manual = mlp.down_proj(gate * up)
        print(f"  gate_proj(x) -> gelu   {list(gate.shape)}   the 'gate': how much to let through")
        print(f"  up_proj(x)             {list(up.shape)}   the 'content'")
        print(f"  gate * up              {list((gate * up).shape)}   elementwise product")
        print(f"  down_proj(...)         {list(manual.shape)}   back to hidden_size")
        print(f"\n  matches mlp(x): {torch.allclose(manual, mlp(x), atol=1e-6)}")

    print("\n  A classic FFN is down(gelu(up(x))): two matrices. GeGLU splits the")
    print("  expansion in two and multiplies them, so the network can *modulate* one")
    print("  projection with another instead of just thresholding it. Costs 50% more")
    print("  parameters at the same intermediate_size, and consistently wins per FLOP")
    print("  (Shazeer, 'GLU Variants Improve Transformer').")

    print()
    print("=" * 74)
    print("4. WHAT THE GATE ACTUALLY DOES")
    print("=" * 74)
    with torch.no_grad():
        gate_values = nn.functional.gelu(mlp.gate_proj(x), approximate="tanh").flatten()
    near_zero = (gate_values.abs() < 0.05).float().mean()
    print(f"  fraction of gate values within +-0.05 of 0: {near_zero:.1%}")
    print("  Those channels are switched OFF for this token: the corresponding entries")
    print("  of up_proj(x) never reach down_proj. Different tokens switch off different")
    print("  channels, which is how one shared FFN behaves differently per token.")

    print()
    print("=" * 74)
    print("5. GELU, AND WHY approximate='tanh'")
    print("=" * 74)
    print("  Exact GELU needs erf(); the tanh form is a cheap approximation:")
    grid = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    exact = nn.functional.gelu(grid)
    approximate = nn.functional.gelu(grid, approximate="tanh")
    print(f"    x        {[f'{v:+.1f}' for v in grid.tolist()]}")
    print(f"    gelu     {[f'{v:+.3f}' for v in exact.tolist()]}")
    print(f"    gelu-tanh{[f'{v:+.3f}' for v in approximate.tolist()]}")
    print(f"    max diff {(exact - approximate).abs().max():.2e}")
    print("  We use the tanh variant because that is what the checkpoint was trained")
    print("  with. Unlike ReLU, GELU is smooth and passes small negative values")
    print("  through, so a channel is never completely dead.")

    print()
    print("=" * 74)
    print("6. EXERCISES")
    print("=" * 74)
    print("  a) Count the parameters of GemmaMLP and compare with a 2-matrix FFN at the")
    print("     same intermediate_size. Then find the intermediate_size that makes them")
    print("     equal (hint: 2/3).")
    print("  b) Swap gate_proj and up_proj in the forward. Does the model still train?")
    print("     Does it still load the checkpoint correctly? Two different questions.")
    print("  c) Print the norm of `residual` and of the branch output at each of the 18")
    print("     layers with real weights. Which one dominates, and what does that tell")
    print("     you about how much each layer changes the stream?")


if __name__ == "__main__":
    main()
