"""Episode 01 -- make sure your machine is ready for the rest of the playlist.

Run:  python check_env.py
"""

import importlib
import platform
import sys

REQUIRED = [
    ("torch", "the only deep learning dependency: we build every layer by hand"),
    ("numpy", "image preprocessing in the PaliGemma processor"),
    ("PIL", "loading the input image (pillow)"),
    ("safetensors", "reading the downloaded PaliGemma weights"),
    ("transformers", "we only use it for the tokenizer, nothing else"),
    ("fire", "turns inference.py into a CLI"),
]


def main() -> int:
    print(f"Python  : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    missing = []
    for name, why in REQUIRED:
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            print(f"MISSING : {name:<13} -- {why}")
            continue
        version = getattr(module, "__version__", "?")
        print(f"OK      : {name:<13} {version:<10} -- {why}")

    if missing:
        print("\nInstall what is missing with:")
        print("    pip install -r requirements.txt")
        return 1

    import torch

    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = "mps (Apple Silicon)"
    else:
        device = "cpu"
    print(f"\nInference device: {device}")
    if device == "cpu":
        print("CPU is fine for this playlist. Generating ~20 tokens with the real")
        print("3B weights takes a couple of minutes; every demo in this repo is instant.")

    # A 224x224 image with 16x16 patches is the setup we use all playlist long.
    image_size, patch_size = 224, 16
    print(f"\nSanity check: a {image_size}x{image_size} image cut into {patch_size}x{patch_size} patches")
    print(f"  -> {(image_size // patch_size) ** 2} patches, i.e. 256 image tokens for paligemma-3b-pt-224")
    return 0


if __name__ == "__main__":
    sys.exit(main())
