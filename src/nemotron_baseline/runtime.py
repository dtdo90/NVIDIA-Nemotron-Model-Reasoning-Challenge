from __future__ import annotations

import os


def disable_transformers_vision_imports() -> None:
    """Avoid optional torchvision imports in text-only training/eval scripts."""
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        import transformers.utils as transformers_utils  # type: ignore
        import transformers.utils.import_utils as import_utils  # type: ignore
    except ImportError:
        return

    def unavailable(*args, **kwargs):
        return False

    for name in ("is_torchvision_available", "is_torchvision_v2_available"):
        original = getattr(import_utils, name, None)
        if hasattr(original, "cache_clear"):
            original.cache_clear()
        setattr(import_utils, name, unavailable)
        setattr(transformers_utils, name, unavailable)


def check_nemotron_runtime_dependencies() -> None:
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "The Nemotron HF model requires mamba-ssm. Install it after PyTorch is installed:\n"
            "  pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm"
        ) from exc
