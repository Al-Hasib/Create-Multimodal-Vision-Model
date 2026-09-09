"""Episode 11 -- what the processor does to your image and your prompt.

The real processor needs the PaliGemma tokenizer from Hugging Face. So that this
check runs offline, we plug in a minimal fake tokenizer that behaves like the real
one for the parts the processor touches. The image pipeline is the real thing.

Run:  python check_processor.py
"""

import numpy as np
import torch
from PIL import Image

from processing_paligemma import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PaliGemmaProcessor,
    add_image_tokens_to_prompt,
    process_images,
)


class FakeTokenizer:
    """Just enough of a tokenizer to drive PaliGemmaProcessor offline."""

    bos_token = "<bos>"
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.vocab = {"<bos>": 2, "<eos>": 1, "\n": 108}
        self.next_id = 256_000  # where PaliGemma's extra tokens start
        self.add_bos_token = True
        self.add_eos_token = True

    def add_special_tokens(self, tokens_to_add):
        self.add_tokens(tokens_to_add["additional_special_tokens"])

    def add_tokens(self, tokens):
        for token in tokens:
            self.vocab.setdefault(token, self.next_id)
            if self.vocab[token] == self.next_id:
                self.next_id += 1

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        # A word-level stand-in for SentencePiece: known tokens keep their id,
        # unknown words get a stable hash-based id in the text range.
        all_ids = []
        for text in texts:
            ids = []
            rest = text
            for token, token_id in sorted(self.vocab.items(), key=lambda kv: -len(kv[0])):
                rest = rest.replace(token, f" \x00{token_id}\x00 ")
            for piece in rest.split():
                if piece.startswith("\x00"):
                    ids.append(int(piece.strip("\x00")))
                else:
                    ids.append(abs(hash(piece)) % 200_000)
            all_ids.append(ids)
        input_ids = torch.tensor(all_ids)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def main() -> None:
    print("=" * 74)
    print("1. THE PROMPT FORMAT")
    print("=" * 74)
    formatted = add_image_tokens_to_prompt(
        prefix_prompt="this building is", bos_token="<bos>", image_seq_len=4, image_token="<image>"
    )
    print(f"  {formatted!r}")
    print("\n  Order matters and it is unusual: IMAGE TOKENS FIRST, then <bos>, then")
    print("  your text, then a newline.")
    print("    - the image tokens are placeholders; Episode 13 overwrites their")
    print("      embeddings with the projected patch embeddings")
    print("    - <bos> sits *after* the image, so the text stream starts there")
    print("    - the trailing '\\n' is part of the training format. Leave it out and")
    print("      the model behaves noticeably worse")
    print("    - no <eos>: we are prefilling a prompt, not scoring a full sequence")

    print()
    print("=" * 74)
    print("2. THE IMAGE PIPELINE: resize -> rescale -> normalize -> transpose")
    print("=" * 74)
    # A 640x480 gradient image, so we can see the resize happen.
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)[None, :]
    raw[:, :, 1] = np.linspace(0, 255, 480, dtype=np.uint8)[:, None]
    raw[:, :, 2] = 128
    image = Image.fromarray(raw)
    print(f"  input PIL image      : size {image.size} (width, height), mode {image.mode}")

    processed = process_images(
        [image],
        size=(224, 224),
        resample=Image.Resampling.BICUBIC,
        rescale_factor=1 / 255.0,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
    )[0]
    print(f"  after process_images : shape {processed.shape}  [Channels, Height, Width]")
    print(f"  value range          : [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"  mean {IMAGENET_STANDARD_MEAN} and std {IMAGENET_STANDARD_STD} map [0, 1] -> [-1, 1]")
    print("  Note this is NOT the ImageNet mean/std you may know (0.485, 0.456, ...):")
    print("  SigLIP was trained with a plain 0.5/0.5, so we must match it exactly.")
    print("  Also note the aspect ratio is *not* preserved -- 640x480 is squashed")
    print("  into a square, which is what the checkpoint expects.")

    print()
    print("=" * 74)
    print("3. THE FULL PROCESSOR CALL")
    print("=" * 74)
    processor = PaliGemmaProcessor(FakeTokenizer(), num_image_tokens=256, image_size=224)
    print(f"  IMAGE_TOKEN id      : {processor.image_token_id}")
    print(f"  image_seq_length    : {processor.image_seq_length}")
    print("  The processor also registers 1024 <locNNNN> tokens (bounding boxes) and")
    print("  128 <segNNN> tokens (segmentation masks): PaliGemma emits detection and")
    print("  segmentation results as ordinary text tokens.")

    inputs = processor(text=["this building is"], images=[image])
    print("\n  returned dict:")
    for key, value in inputs.items():
        print(f"    {key:<15} {list(value.shape)}  {value.dtype}")
    input_ids = inputs["input_ids"][0]
    print(f"\n  first 4 ids  : {input_ids[:4].tolist()}   <- image placeholders")
    print(f"  ids 254..260 : {input_ids[254:261].tolist()}   <- image ends, then <bos>, then text")
    print(f"  image tokens : {(input_ids == processor.image_token_id).sum().item()} of {len(input_ids)}")

    print()
    print("=" * 74)
    print("4. WHY RIGHT PADDING, AND WHY WE ONLY ALLOW ONE IMAGE")
    print("=" * 74)
    print("  utils.load_hf_model asserts padding_side == 'right' and the processor")
    print("  asserts a single image / single prompt. During generation we append the")
    print("  new token at the end of the sequence, so any padding must be *after* the")
    print("  prompt; and with one sample there is no padding at all, which is what lets")
    print("  Episode 15 get away with an all-zero attention mask.")

    print()
    print("=" * 74)
    print("5. EXERCISES")
    print("=" * 74)
    print("  a) Remove the trailing '\\n' from add_image_tokens_to_prompt and compare")
    print("     the generated caption once you have the real weights (Episode 20).")
    print("  b) Change num_image_tokens to 128 and follow the crash: which assert or")
    print("     which shape mismatch catches you first, and why?")
    print("  c) Swap BICUBIC for NEAREST and measure the mean absolute difference of")
    print("     the resulting tensor.")


if __name__ == "__main__":
    main()
