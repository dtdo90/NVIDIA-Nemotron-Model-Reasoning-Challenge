from __future__ import annotations

import os
import importlib


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
    required_modules = [
        ("einops", "einops"),
        ("causal_conv1d", "causal-conv1d"),
        ("mamba_ssm", "mamba-ssm"),
    ]
    for module_name, package_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise SystemExit(
                "The Nemotron HF model requires mamba-ssm and its runtime helpers.\n"
                f"Missing import: {module_name}\n"
                f"Install package: {package_name}\n"
                f"Import error: {exc}\n"
                "Install them after PyTorch is installed:\n"
                "  pip install --no-build-isolation --no-deps -r requirements-nemotron.txt"
            ) from exc
        except Exception as exc:
            # Compiled extensions can be present but ABI-incompatible with the
            # active PyTorch/CUDA runtime. Surface that as a setup issue too.
            raise SystemExit(
                "The Nemotron HF model requires mamba-ssm and its runtime helpers.\n"
                f"Failed import: {module_name}\n"
                f"Package: {package_name}\n"
                f"Import error: {exc}\n"
                "Reinstall these packages after PyTorch is installed:\n"
                "  pip install --force-reinstall --no-build-isolation --no-deps -r requirements-nemotron.txt"
            ) from exc
