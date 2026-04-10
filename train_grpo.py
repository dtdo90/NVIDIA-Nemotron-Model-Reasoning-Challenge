#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import (
    infer_category,
    load_split_assignments,
    select_ids_for_splits,
    summarize_categories,
    summarize_split_assignments,
)
from nemotron_baseline.prompts import build_messages
from nemotron_baseline.rewards import (
    accuracy_reward,
    final_line_reward,
    single_box_reward,
)


@dataclass(frozen=True)
class RLExample:
    id: str
    prompt: str
    answer: str
    category: str
    label: str | None = None


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if not config_path:
        return {}
    with Path(config_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("Config file must contain a JSON object.")
    return payload


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    defaults = load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Run GRPO starting from an SFT Nemotron LoRA adapter."
    )
    parser.add_argument("--config", default=bootstrap_args.config)
    parser.add_argument("--train-csv", default=defaults.get("train_csv", "data/train.csv"))
    parser.add_argument("--split-csv", default=defaults.get("split_csv", "data/splits_70_15_15.csv"))
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=defaults.get("train_splits", ["grpo_train"]),
        help="Named splits from --split-csv to use for GRPO training.",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults.get("output_dir", "outputs/grpo_stage2"),
    )
    parser.add_argument("--sft-adapter-dir", default=defaults.get("sft_adapter_dir"))
    parser.add_argument("--model-path", default=defaults.get("model_path"))
    parser.add_argument(
        "--kaggle-model-handle",
        default=defaults.get(
            "kaggle_model_handle",
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        ),
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--max-train-samples", type=int, default=defaults.get("max_train_samples"))
    parser.add_argument("--max-completion-length", type=int, default=defaults.get("max_completion_length", 1024))
    parser.add_argument("--num-epochs", type=float, default=defaults.get("num_epochs", 1.0))
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=defaults.get("per_device_train_batch_size", 1),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.get("gradient_accumulation_steps", 4),
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.get("learning_rate", 5e-6))
    parser.add_argument("--num-generations", type=int, default=defaults.get("num_generations", 4))
    parser.add_argument("--temperature", type=float, default=defaults.get("temperature", 1.0))
    parser.add_argument("--top-p", type=float, default=defaults.get("top_p", 1.0))
    parser.add_argument("--top-k", type=int, default=defaults.get("top_k", 0))
    parser.add_argument("--min-p", type=float, default=defaults.get("min_p"))
    parser.add_argument("--repetition-penalty", type=float, default=defaults.get("repetition_penalty", 1.0))
    parser.add_argument("--beta", type=float, default=defaults.get("beta", 0.0))
    parser.add_argument("--num-iterations", type=int, default=defaults.get("num_iterations", 1))
    parser.add_argument("--epsilon", type=float, default=defaults.get("epsilon", 0.2))
    parser.add_argument("--loss-type", default=defaults.get("loss_type", "dr_grpo"))
    parser.add_argument("--scale-rewards", default=defaults.get("scale_rewards", "none"))
    parser.add_argument("--logging-steps", type=int, default=defaults.get("logging_steps", 10))
    parser.add_argument("--save-tokenizer", action="store_true", default=bool(defaults.get("save_tokenizer", False)))
    parser.add_argument("--disable-kaggle-triton-fixes", action="store_true")
    return parser.parse_args(remaining)


def require_training_dependencies():
    try:
        import torch  # type: ignore
        import torch.nn.functional as torch_f  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing GRPO dependencies. Install transformers, datasets, peft, trl, and torch."
        ) from exc

    return {
        "torch": torch,
        "torch_f": torch_f,
        "Dataset": Dataset,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def apply_kaggle_triton_fixes(torch, torch_f) -> None:
    def _pure_rmsnorm_fn(
        x,
        weight,
        bias=None,
        z=None,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        upcast=True,
    ):
        del group_size, norm_before_gate
        dtype = x.dtype
        if upcast:
            x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        out = x_normed * weight.float()
        if bias is not None:
            out = out + bias.float()
        if z is not None:
            out = out * torch_f.silu(z.float())
        return out.to(dtype)

    for module in list(sys.modules.values()):
        if hasattr(module, "rmsnorm_fn"):
            module.rmsnorm_fn = _pure_rmsnorm_fn

    src = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
        "triton/backends/nvidia/bin/ptxas-blackwell"
    )
    dst = "/tmp/ptxas-blackwell"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        import triton.backends.nvidia as nv_backend  # type: ignore

        src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for name in os.listdir(dst_bin):
            file_path = os.path.join(dst_bin, name)
            if os.path.isfile(file_path):
                os.chmod(
                    file_path,
                    os.stat(file_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
        os.environ["TRITON_PTXAS_PATH"] = dst
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst

        import triton.backends.nvidia.compiler as nv_compiler  # type: ignore

        nv_compiler.get_ptxas_version = lambda arch: "12.0"


def disable_nemotron_fast_path() -> None:
    for name, module in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False


def resolve_model_path(args: argparse.Namespace, kagglehub_module) -> str:
    if args.model_path:
        return args.model_path
    if kagglehub_module is None:
        raise SystemExit(
            "No --model-path was provided and kagglehub is not installed. "
            "Either install kagglehub or pass a local base-model path."
        )
    return kagglehub_module.model_download(args.kaggle_model_handle)


def load_training_examples(csv_path: str) -> list[RLExample]:
    examples: list[RLExample] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            examples.append(
                RLExample(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    category=infer_category(row["prompt"]),
                    label=row.get("label"),
                )
            )
    return examples


def build_dataset(dataset_cls, examples: list[RLExample]):
    rows = [
        {
            "id": example.id,
            "prompt": build_messages(example.prompt, None),
            "answer": example.answer,
            "category": example.category,
            "label": example.label,
        }
        for example in examples
    ]
    return dataset_cls.from_list(rows)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_trainable_adapter(model) -> None:
    for name, parameter in model.named_parameters():
        if "lora_" in name or "adapter" in name:
            parameter.requires_grad = True
    model.train()


def main() -> None:
    args = parse_args()
    if not args.sft_adapter_dir:
        raise SystemExit("--sft-adapter-dir is required.")
    if args.num_generations < 1:
        raise SystemExit("--num-generations must be at least 1.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    if effective_batch % args.num_generations != 0:
        raise SystemExit(
            "The effective batch size "
            f"({effective_batch} = WORLD_SIZE {world_size} * per_device_train_batch_size "
            f"{args.per_device_train_batch_size} * gradient_accumulation_steps "
            f"{args.gradient_accumulation_steps}) must be divisible by num_generations "
            f"({args.num_generations})."
        )

    deps = require_training_dependencies()
    try:
        import kagglehub  # type: ignore
    except ImportError:
        kagglehub = None

    if not args.disable_kaggle_triton_fixes:
        apply_kaggle_triton_fixes(deps["torch"], deps["torch_f"])

    model_path = resolve_model_path(args, kagglehub)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    split_assignments = load_split_assignments(args.split_csv)
    selected_ids = select_ids_for_splits(split_assignments, args.train_splits)
    examples = load_training_examples(args.train_csv)
    train_examples = [example for example in examples if example.id in selected_ids]
    if not train_examples:
        raise SystemExit(
            f"No GRPO training examples matched splits {args.train_splits!r} in {args.split_csv!r}."
        )
    if args.max_train_samples is not None:
        train_examples = train_examples[: args.max_train_samples]

    tokenizer = deps["AutoTokenizer"].from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = deps["AutoModelForCausalLM"].from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=deps["torch"].bfloat16,
    )
    base_model.gradient_checkpointing_enable()
    disable_nemotron_fast_path()

    try:
        model = deps["PeftModel"].from_pretrained(
            base_model,
            args.sft_adapter_dir,
            is_trainable=True,
        )
    except TypeError:
        model = deps["PeftModel"].from_pretrained(base_model, args.sft_adapter_dir)
        ensure_trainable_adapter(model)
    model.train()

    train_dataset = build_dataset(deps["Dataset"], train_examples)

    trainer_config = deps["GRPOConfig"](
        output_dir=str(output_dir / "trainer_state"),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        bf16=True,
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.0,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        chat_template_kwargs=None,
        log_completions=False,
    )

    trainer = deps["GRPOTrainer"](
        model=model,
        reward_funcs=[accuracy_reward, single_box_reward, final_line_reward],
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=trainer_config,
    )
    trainer.train()

    trainer.model.save_pretrained(adapter_dir)
    if args.save_tokenizer:
        tokenizer.save_pretrained(adapter_dir)

    submission_path = output_dir / "submission.zip"
    with zipfile.ZipFile(submission_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in sorted(adapter_dir.iterdir()):
            zip_handle.write(file_path, file_path.name)

    config_payload = {
        "train_csv": str(Path(args.train_csv).resolve()),
        "split_csv": str(Path(args.split_csv).resolve()),
        "train_splits": args.train_splits,
        "output_dir": str(output_dir.resolve()),
        "adapter_dir": str(adapter_dir.resolve()),
        "submission_zip": str(submission_path.resolve()),
        "sft_adapter_dir": str(Path(args.sft_adapter_dir).resolve()),
        "model_path": model_path,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_completion_length": args.max_completion_length,
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "repetition_penalty": args.repetition_penalty,
        "beta": args.beta,
        "num_iterations": args.num_iterations,
        "epsilon": args.epsilon,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "save_tokenizer": args.save_tokenizer,
    }
    dataset_summary = {
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "train_category_counts": summarize_categories(train_examples),
        "split_counts": summarize_split_assignments(split_assignments),
    }

    write_json(output_dir / "run_config.json", config_payload)
    write_json(output_dir / "dataset_summary.json", dataset_summary)

    print(f"Model path: {model_path}")
    print(f"SFT adapter: {Path(args.sft_adapter_dir).resolve()}")
    print(f"GRPO rows: {len(train_examples)}")
    print(f"Train splits: {args.train_splits}")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Submission zip: {submission_path}")


if __name__ == "__main__":
    main()
