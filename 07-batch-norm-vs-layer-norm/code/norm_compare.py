"""Episode 07 -- batch normalization vs layer normalization.

Both fight the same problem (activations drifting into a bad range, so gradients
either vanish or explode and every layer keeps chasing a moving input distribution).
They differ in *which* numbers they average over -- and that single difference is why
transformers use LayerNorm everywhere.

Run:  python norm_compare.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

BATCH_SIZE, SEQ_LEN, FEATURES = 4, 3, 6


def main() -> None:
    # A batch of token embeddings, the shape every norm in this repo sees.
    x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURES) * 3 + 5
    print(f"input x: [Batch_Size, Seq_Len, Features] = {list(x.shape)}")
    print(f"  raw statistics: mean {x.mean():+.3f}, std {x.std():.3f}")

    print()
    print("=" * 74)
    print("1. WHAT EACH ONE AVERAGES OVER")
    print("=" * 74)
    print("  BatchNorm : one mean/variance PER FEATURE, computed across the batch")
    print("              (and across the sequence). Statistics of feature j depend on")
    print("              every *other sample* in the batch.")
    print("  LayerNorm : one mean/variance PER TOKEN, computed across the features of")
    print("              that token alone. No sample ever looks at another sample.")

    # LayerNorm, by hand, over the last dimension only.
    ln_mean = x.mean(dim=-1, keepdim=True)
    ln_var = x.var(dim=-1, keepdim=True, unbiased=False)
    manual_ln = (x - ln_mean) / torch.sqrt(ln_var + 1e-5)
    torch_ln = nn.LayerNorm(FEATURES, eps=1e-5)(x)
    print(f"\n  manual LayerNorm vs nn.LayerNorm : max diff {(manual_ln - torch_ln).abs().max():.3e}")

    # BatchNorm1d expects [Batch, Features, Seq_Len], so we transpose.
    bn_mean = x.mean(dim=(0, 1))
    bn_var = x.var(dim=(0, 1), unbiased=False)
    manual_bn = (x - bn_mean) / torch.sqrt(bn_var + 1e-5)
    torch_bn = nn.BatchNorm1d(FEATURES, eps=1e-5)(x.transpose(1, 2)).transpose(1, 2)
    print(f"  manual BatchNorm vs nn.BatchNorm1d: max diff {(manual_bn - torch_bn).abs().max():.3e}")

    print(f"\n  after LayerNorm : mean per token   = {manual_ln.mean(dim=-1).flatten()[:4].tolist()}")
    print(f"  after BatchNorm : mean per feature = {manual_bn.mean(dim=(0, 1)).tolist()}")

    print()
    print("=" * 74)
    print("2. THE KILLER: BatchNorm COUPLES YOUR SAMPLES")
    print("=" * 74)
    # Change ONE sample, then look at what happened to a different sample.
    x2 = x.clone()
    x2[3] += 50.0  # an outlier lands in the batch

    bn_before = manual_bn[0, 0]
    m, v = x2.mean(dim=(0, 1)), x2.var(dim=(0, 1), unbiased=False)
    bn_after = ((x2 - m) / torch.sqrt(v + 1e-5))[0, 0]
    ln_after = ((x2 - x2.mean(-1, keepdim=True)) / torch.sqrt(x2.var(-1, keepdim=True, unbiased=False) + 1e-5))[0, 0]

    print("  We modify sample 3 only, then inspect sample 0, token 0:")
    print(f"    BatchNorm before : {[round(v, 3) for v in bn_before.tolist()]}")
    print(f"    BatchNorm after  : {[round(v, 3) for v in bn_after.tolist()]}   <- CHANGED")
    print(f"    LayerNorm before : {[round(v, 3) for v in manual_ln[0, 0].tolist()]}")
    print(f"    LayerNorm after  : {[round(v, 3) for v in ln_after.tolist()]}   <- untouched")
    print("\n  Consequences of that coupling:")
    print("    - small batches give noisy statistics (batch size 1 has zero variance)")
    print("    - training and inference behave differently, so BatchNorm needs running")
    print("      averages and a train/eval switch")
    print("    - variable-length sequences pollute the per-feature statistics with padding")
    print("    - multi-GPU training needs the batch statistics synchronised across devices")

    print()
    print("=" * 74)
    print("3. BATCH SIZE 1: BatchNorm HAS NOTHING TO WORK WITH")
    print("=" * 74)
    single = x[:1]
    print(f"  single sample, per-feature variance across batch = "
          f"{single.var(dim=0, unbiased=False).flatten()[:3].tolist()} (per position: all zeros)")
    single_ln = nn.LayerNorm(FEATURES)(single)
    print(f"  LayerNorm still works fine: output std per token = "
          f"{[round(v, 3) for v in single_ln.std(dim=-1).flatten().tolist()]}")
    print("  Autoregressive generation *is* batch size 1 with one token at a time.")

    print()
    print("=" * 74)
    print("4. gamma AND beta: NORMALIZATION IS NOT A STRAITJACKET")
    print("=" * 74)
    layer_norm = nn.LayerNorm(FEATURES)
    print(f"  learnable weight (gamma) : {list(layer_norm.weight.shape)}, init {layer_norm.weight[0].item():.1f}")
    print(f"  learnable bias   (beta)  : {list(layer_norm.bias.shape)}, init {layer_norm.bias[0].item():.1f}")
    print("  Zero mean / unit variance is only the starting point: the model can learn")
    print("  to scale and shift each feature back to whatever range it needs.")

    print()
    print("=" * 74)
    print("5. WHERE THIS SHOWS UP IN OUR MODEL")
    print("=" * 74)
    print("  SigLIP encoder layer : nn.LayerNorm, pre-norm (norm -> sublayer -> add).")
    print("                         Post-norm was the original transformer; pre-norm")
    print("                         keeps a clean residual path and trains far more")
    print("                         stably at depth. See Episode 08.")
    print("  Gemma decoder layer  : RMSNorm -- LayerNorm minus the mean subtraction")
    print("                         and minus the bias. Cheaper, works just as well.")
    print("                         See Episode 16.")


if __name__ == "__main__":
    main()
