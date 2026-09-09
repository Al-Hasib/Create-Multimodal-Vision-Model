# Episode 10 — Coding Multi-Head Attention (and finishing SigLIP)

> **Video chapters:** `~01:50:00 – 02:18:30` (Multi-Head Attention coding · Coding SigLip `02:15:40`)
> **Target runtime:** 20–25 min · **Code:** [`code/modeling_siglip.py`](code/modeling_siglip.py), [`code/check_siglip.py`](code/check_siglip.py)

Replace the placeholder with the real `SiglipAttention`. After this episode the vision
tower is **finished** — 412 M parameters, ready for the real weights.

## `__init__`

```python
class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5      # == 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
```

Four square projections, each `[embed_dim, embed_dim]`, **with** biases (SigLIP has them;
Gemma will not). Note `head_dim = embed_dim // num_heads` — here the two are coupled, and
they must divide evenly. Gemma decouples them (Episode 12).

Q, K and V are separate `Linear`s rather than one fused `[embed_dim, 3*embed_dim]` layer,
because that is how the checkpoint stores them. Fusing would be faster and would not load.

`self.scale` as a multiply is the same as dividing by `sqrt(head_dim)`, and cheaper.

## `forward`

```python
def forward(self, hidden_states):
    batch_size, seq_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)   # [B, Num_Patches, Embed_Dim]
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # [B, Num_Patches, Embed_Dim] -> [B, Num_Heads, Num_Patches, Head_Dim]
    query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Q @ K^T / sqrt(d_k)  ->  [B, Num_Heads, Num_Patches, Num_Patches]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

    if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
        raise ValueError(...)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)   # [B, Num_Heads, Num_Patches, Head_Dim]

    attn_output = attn_output.transpose(1, 2).contiguous()   # [B, Num_Patches, Num_Heads, Head_Dim]
    attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights
```

Everything here was derived in Episode 09. The points worth saying out loud while typing:

- **`transpose(2, 3)`, not `.t()` or `transpose(0, 1)`.** We are transposing the last two
  dimensions of a 4-D tensor; the batch and head dimensions must stay put.
- **`dtype=torch.float32` in the softmax**, then `.to(query_states.dtype)`. Episode 03.
- **`training=self.training`** makes dropout a no-op in `eval()`. Since
  `attention_dropout=0.0` it is a no-op regardless, but this is how you would write it for
  training, and it costs nothing.
- **The two shape assertions.** Cheap, and they catch precisely the class of bug that
  otherwise runs silently for 27 layers. Keep them.
- **No mask parameter at all** — this is the vision tower (Episode 05).

## Verifying it, three ways

The check script does all three, which is more valuable than reading the code again:

**1. Against PyTorch's fused kernel.** Reuse our own projections, feed Q/K/V to
`F.scaled_dot_product_attention`, apply our `out_proj`, and compare: `max diff ≈ 1e-7`.
Same maths. PyTorch's version just never materialises the `[Seq, Seq]` matrix, which is why
you would use it in production and not for learning.

**2. Information now flows.** The Episode 08 experiment, re-run: perturb patch 0, and every
other output token moves (smallest movement ≈ 0.07 instead of exactly 0). Attention is the
only thing that changed.

**3. The real tower loads.** Instantiate the So400m config and count:

```
hidden_size 1152, 27 layers, 16 heads -> head_dim 72
parameters per encoder layer :   15,239,504
parameters in the whole tower:  412,442,352   (~412M, hence "So400m")
[1, 3, 224, 224] -> [1, 256, 1152]
```

256 vectors of 1152 dimensions per image. Episode 13 projects those to Gemma's 2048.

## Run it

```bash
cd 10-multi-head-attention-code/code
python check_siglip.py
```

## Key takeaways

- Four separate square `Linear`s with bias, matching the checkpoint's layout.
- `view(B, Seq, H, D).transpose(1, 2)` in, `transpose(1, 2).contiguous().reshape(...)` out.
- Softmax in float32 on `dim=-1`; scale by `head_dim**-0.5`.
- Shape assertions are worth their line count.
- Our implementation matches `F.scaled_dot_product_attention` to float precision.
- **The vision tower is done.** 412 M parameters, 256 tokens per image.

## Gotchas

- **`view` before `transpose`, in that order.** `transpose` then `view` reinterprets the
  wrong memory layout — it may not even error, and the heads end up scrambled across
  positions.
- **`reshape` without `.contiguous()`** works (reshape copies when it must), but `view`
  raises. Knowing which is which saves an afternoon.
- **Fusing q/k/v into one `Linear`** breaks checkpoint loading.
- **Forgetting `out_proj`** returns concatenated heads that never interact.
- **Dropping `* self.scale`** looks harmless on small `head_dim` and gets much worse as the
  model widens.

## Exercises

1. Remove `* self.scale` and print `attn_weights.max()` for `head_dim` 8, 72 and 512.
2. Set `num_attention_heads=1152` (`head_dim=1`). What still runs? What has the model lost?
   Now set it to 5 and read the error.
3. Return the concatenated heads without `out_proj` and explain, referring to the residual
   stream, why the heads can no longer combine their findings.
4. Feed the same image twice in one batch and confirm the two outputs are identical —
   attention must never leak across the batch dimension.
5. Swap our attention for `F.scaled_dot_product_attention` inside `SiglipAttention` and
   compare peak memory for `seq_len=1024` using `torch.profiler` or `tracemalloc`.

**Previous:** [Episode 09](../09-multi-head-attention-theory/) ·
**Next:** [Episode 11 — PaliGemma architecture & input processor](../11-paligemma-input-processor/)
