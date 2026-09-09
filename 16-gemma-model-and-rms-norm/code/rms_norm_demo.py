"""Episode 16 -- RMSNorm: LayerNorm minus the mean, minus the bias.

Run:  python rms_norm_demo.py
"""

import torch
import torch.nn as nn

from modeling_gemma import GemmaRMSNorm

torch.manual_seed(0)


def main() -> None:
    features = 8
    x = torch.randn(2, 3, features) * 2 + 5

    print("=" * 74)
    print("1. THE TWO FORMULAS SIDE BY SIDE")
    print("=" * 74)
    print("  LayerNorm : (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta")
    print("  RMSNorm   :  x           / sqrt(mean(x^2) + eps) * gamma")
    print("\n  Dropped: the mean subtraction (re-centering) and the bias.")
    print("  Kept    : the re-scaling, which is the part that actually stabilises")
    print("            training. The 2019 RMSNorm paper's claim, and it held up.")

    rms = GemmaRMSNorm(features)
    with torch.no_grad():
        ours = rms(x)
        manual = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms.eps) * (1.0 + rms.weight)
    print(f"\n  our GemmaRMSNorm vs the formula: max diff {(ours - manual).abs().max():.3e}")

    print()
    print("=" * 74)
    print("2. WHAT EACH ONE DOES TO A VECTOR")
    print("=" * 74)
    layer_norm = nn.LayerNorm(features, elementwise_affine=False)
    with torch.no_grad():
        ln = layer_norm(x)
        rms_only = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    print(f"  input       : mean {x[0, 0].mean():+.3f}  rms {x[0, 0].pow(2).mean().sqrt():.3f}")
    print(f"  LayerNorm   : mean {ln[0, 0].mean():+.3f}  rms {ln[0, 0].pow(2).mean().sqrt():.3f}   <- centered")
    print(f"  RMSNorm     : mean {rms_only[0, 0].mean():+.3f}  rms {rms_only[0, 0].pow(2).mean().sqrt():.3f}"
          f"   <- NOT centered, only rescaled")
    print("\n  RMSNorm preserves the direction of the vector exactly and only fixes its")
    print("  length. LayerNorm also shifts it towards the origin.")

    print()
    print("=" * 74)
    print("3. THE (1.0 + self.weight) QUIRK")
    print("=" * 74)
    print(f"  self.weight is initialised to ZEROS: {rms.weight[:4].tolist()}")
    print("  and the forward computes `output * (1.0 + self.weight)`.")
    print("  So at initialisation the scale is 1.0, exactly like LayerNorm's")
    print("  weight-initialised-to-ones -- Gemma just stores (gamma - 1) instead of")
    print("  gamma. If you write `output * self.weight` with a Gemma checkpoint, every")
    print("  normalized activation gets multiplied by roughly 0 and the model outputs")
    print("  garbage. It is one of the classic 'why is my re-implementation broken'")
    print("  bugs, so it is worth remembering.")

    print()
    print("=" * 74)
    print("4. THE float() DANCE")
    print("=" * 74)
    print("  Our forward is:")
    print("      output = self._norm(x.float())")
    print("      output = output * (1.0 + self.weight.float())")
    print("      return output.type_as(x)")
    print("  x.pow(2) in float16 overflows around x > 256, so the statistics are always")
    print("  computed in float32 and cast back afterwards.")
    print("  Also note Gemma computes (x * w) in float32 and casts at the very end,")
    print("  while Llama casts x first and then multiplies -- a real, if tiny,")
    print("  numerical difference between the two families (see HF PR #29402).")
    x16 = torch.tensor([[[300.0] * features]], dtype=torch.float16)
    naive = (x16.pow(2).mean(-1))
    print(f"\n  (300^2 in float16).mean() = {naive.item()}   <- inf, hence the .float()")

    print()
    print("=" * 74)
    print("5. WHY BOTHER? BECAUSE IT RUNS 18 x 2 + 1 TIMES PER TOKEN")
    print("=" * 74)
    print("  Per decoder layer: input_layernorm + post_attention_layernorm, plus one")
    print("  final norm at the end of the stack. With 18 layers that is 37 normalizations")
    print("  for every single token. Removing the mean pass and the bias add saves a")
    print("  reduction and a tensor read each time -- small in FLOPs, real in memory")
    print("  traffic, which is what decoding is bound by (Episode 14).")


if __name__ == "__main__":
    main()
