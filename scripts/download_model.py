#!/usr/bin/env python3
"""Download and quantize multilingual-e5-small ONNX model for local embedding.

Downloads from HuggingFace:
- intfloat/multilingual-e5-small ONNX model (FP32)
- Tokenizer files

Then applies dynamic INT8 quantization for smaller size and faster inference.

Output directory: ~/.claude/models/multilingual-e5-small/
"""
import os
import sys
import urllib.request
import shutil
from pathlib import Path


# HuggingFace base URL for the model
HF_BASE = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main"

# Files to download
MODEL_FILES = {
    "onnx/model.onnx": "model_fp32.onnx",
}

TOKENIZER_FILES = [
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]

# Output directory
DEFAULT_MODEL_DIR = Path.home() / ".claude" / "models" / "multilingual-e5-small"


def get_model_dir() -> Path:
    """Get model directory, respecting CLAUDE_REFLECT_MODEL_DIR env var."""
    custom = os.environ.get("CLAUDE_REFLECT_MODEL_DIR")
    if custom:
        return Path(custom)
    return DEFAULT_MODEL_DIR


def download_file(url: str, dest: Path, description: str = "") -> None:
    """Download a file with progress indication."""
    desc = description or dest.name
    print(f"  Downloading {desc}...", end="", flush=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "claude-reflect/1.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 8192

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  Downloading {desc}... {pct:.0f}%", end="", flush=True)

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\r  Downloading {desc}... done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"\r  Downloading {desc}... FAILED: {e}")
        if dest.exists():
            dest.unlink()
        raise


def quantize_model(fp32_path: Path, int8_path: Path) -> None:
    """Apply dynamic INT8 quantization to the ONNX model."""
    print("  Quantizing to INT8 (dynamic)...", end="", flush=True)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print(" FAILED")
        print("Error: onnxruntime.quantization not available.")
        print("Install: pip install onnxruntime onnx")
        sys.exit(1)

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    int8_size = int8_path.stat().st_size / (1024 * 1024)
    ratio = int8_size / fp32_size * 100

    print(f" done")
    print(f"    FP32: {fp32_size:.1f} MB -> INT8: {int8_size:.1f} MB ({ratio:.0f}%)")


def main() -> int:
    model_dir = get_model_dir()
    int8_path = model_dir / "model_int8.onnx"

    # Check if already downloaded
    if int8_path.exists() and (model_dir / "tokenizer.json").exists():
        print(f"Model already exists at {model_dir}")
        print("Use --force to re-download.")
        if "--force" not in sys.argv:
            return 0

    print(f"Model directory: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    print("\n[1/3] Downloading ONNX model...")
    fp32_path = model_dir / "model_fp32.onnx"
    for remote_path, local_name in MODEL_FILES.items():
        url = f"{HF_BASE}/{remote_path}"
        dest = model_dir / local_name
        download_file(url, dest, local_name)

    # Download tokenizer files
    print("\n[2/3] Downloading tokenizer files...")
    for filename in TOKENIZER_FILES:
        url = f"{HF_BASE}/{filename}"
        dest = model_dir / filename
        download_file(url, dest)

    # Quantize
    print("\n[3/3] Quantizing model...")
    quantize_model(fp32_path, int8_path)

    # Clean up FP32 model to save space
    if fp32_path.exists() and int8_path.exists():
        fp32_path.unlink()
        print(f"  Removed FP32 model (keeping INT8 only)")

    print(f"\nDone! Model ready at: {model_dir}")
    print(f"  model_int8.onnx: {int8_path.stat().st_size / (1024*1024):.1f} MB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
