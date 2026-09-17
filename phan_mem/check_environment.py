from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    "torch",
    "cv2",
    "numpy",
    "pandas",
    "yaml",
    "tqdm",
    "requests",
    "sklearn",
    "scipy",
    "ultralytics",
]


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    missing = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:
            print(f"[MISSING] {module_name}: {exc}")
            missing.append(module_name)
    if missing:
        print("\nInstall missing packages into this exact interpreter:")
        print(f"\"{sys.executable}\" -m pip install -r requirements.txt")
        raise SystemExit(1)
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU acceleration is not active. For NVIDIA GPUs, install requirements-gpu.txt.")
    print("\nEnvironment is ready.")


if __name__ == "__main__":
    main()
