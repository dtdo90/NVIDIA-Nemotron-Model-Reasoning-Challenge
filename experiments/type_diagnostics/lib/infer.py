from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .common import (
    DATA_DIR,
    QUESTION_TYPES,
    assert_type_dataset_fresh,
    classify_subtype,
    is_eval_eligible,
    load_split_assignments,
    normalize_question_type,
    read_csv_rows,
    select_rows_for_splits,
    type_paths,
    validate_split_assignments,
    write_json,
)

from infer_eval import (  # type: ignore
    MODEL_PATH,
    generate_with_transformers,
    generate_with_vllm,
    resolve_model_path,
)
from nemotron_baseline.metric import score_prediction


@dataclass(frozen=True)
class EvalExample:
    id: str
    prompt: str
    answer: str
    category: str
    diagnostic_subtype: str
    source_mode: str
    source: str


def parse_args(default_question_type: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one diagnostic adapter and report subtype accuracy."
    )
    if default_question_type is None:
        parser.add_argument("--question-type", required=True)
    else:
        parser.set_defaults(question_type=default_question_type)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--kaggle-model-handle",
        default="metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["eval_holdout"],
        help="Diagnostic split names to evaluate. Use both holdouts with: --eval-splits eval_holdout grpo_holdout",
    )
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=7680)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--disable-kaggle-triton-fixes", action="store_true")
    parser.add_argument("--failed-traces-per-subtype", type=int, default=3)
    args = parser.parse_args()
    args.question_type = normalize_question_type(args.question_type)
    paths = type_paths(args.question_type, data_dir=Path(args.data_dir))
    args.adapter_dir = args.adapter_dir or str(paths.output_dir / "adapter")
    args.report_dir = args.report_dir or str(paths.report_dir)
    return args


def load_eval_examples(paths, eval_splits: list[str], max_eval_samples: int | None) -> list[EvalExample]:
    assert_type_dataset_fresh(paths.slug, type_csv=paths.train_csv)
    rows, _ = read_csv_rows(paths.train_csv)
    assignments = load_split_assignments(paths.split_csv)
    validate_split_assignments(rows, assignments, split_csv=paths.split_csv)
    selected_rows = select_rows_for_splits(rows, assignments, eval_splits)
    skipped_ineligible = [row for row in selected_rows if not is_eval_eligible(row)]
    selected_rows = [row for row in selected_rows if is_eval_eligible(row)]
    if max_eval_samples is not None:
        selected_rows = selected_rows[:max_eval_samples]
    if not selected_rows:
        raise SystemExit(
            f"No eval-eligible rows selected for {paths.slug} splits {eval_splits!r} in {paths.split_csv}. "
            f"Skipped eval-ineligible rows: {len(skipped_ineligible)}"
        )

    examples: list[EvalExample] = []
    for row in selected_rows:
        subtype = row.get("diagnostic_subtype") or classify_subtype(row)
        examples.append(
            EvalExample(
                id=row["id"],
                prompt=row["prompt"],
                answer=row["answer"],
                category=row.get("category") or QUESTION_TYPES[paths.slug]["category"],
                diagnostic_subtype=subtype,
                source_mode=row.get("source_mode", "unknown"),
                source=row.get("source", ""),
            )
        )
    return examples


def stats_payload(total: int, correct: int) -> dict[str, int | float]:
    return {
        "total": total,
        "correct": correct,
        "accuracy": float(correct) / float(total) if total else 0.0,
    }


def summarize_by_subtype(examples: list[EvalExample], results) -> dict[str, object]:
    counters: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_source: dict[str, Counter] = defaultdict(Counter)
    for example, result in zip(examples, results):
        stats = counters[example.diagnostic_subtype]
        stats["total"] += 1
        stats["correct"] += 1 if result.correct else 0
        by_source[example.diagnostic_subtype][example.source_mode] += 1

    return {
        subtype: {
            **stats_payload(stats["total"], stats["correct"]),
            "source_modes": dict(sorted(by_source[subtype].items())),
        }
        for subtype, stats in sorted(counters.items())
    }


def write_predictions(path: Path, examples: list[EvalExample], results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for example, result in zip(examples, results):
        lines.append(
            json.dumps(
                {
                    "id": result.example_id,
                    "category": result.category,
                    "diagnostic_subtype": example.diagnostic_subtype,
                    "source_mode": example.source_mode,
                    "source": example.source,
                    "gold_answer": result.gold_answer,
                    "extracted_prediction": result.extracted_prediction,
                    "correct": result.correct,
                    "raw_prediction": result.raw_prediction,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failed_trace_samples(
    report_dir: Path,
    examples: list[EvalExample],
    results,
    *,
    per_subtype: int,
) -> dict[str, int]:
    failed_by_subtype: dict[str, list[tuple[EvalExample, object]]] = defaultdict(list)
    subtype_totals: Counter = Counter()
    subtype_correct: Counter = Counter()
    for example, result in zip(examples, results):
        subtype_totals[example.diagnostic_subtype] += 1
        subtype_correct[example.diagnostic_subtype] += 1 if result.correct else 0
        if not result.correct:
            failed_by_subtype[example.diagnostic_subtype].append((example, result))

    written_counts: dict[str, int] = {}
    failed_root = report_dir / "failed_traces"
    if failed_root.exists():
        shutil.rmtree(failed_root)
    for subtype in sorted(subtype_totals):
        if subtype_correct[subtype] == subtype_totals[subtype]:
            continue
        subtype_dir = failed_root / subtype
        subtype_dir.mkdir(parents=True, exist_ok=True)
        samples = failed_by_subtype.get(subtype, [])[:per_subtype]
        written_counts[subtype] = len(samples)
        for example, result in samples:
            payload = {
                "id": example.id,
                "category": example.category,
                "diagnostic_subtype": example.diagnostic_subtype,
                "source_mode": example.source_mode,
                "source": example.source,
                "gold_answer": result.gold_answer,
                "extracted_prediction": result.extracted_prediction,
                "correct": result.correct,
                "prompt": example.prompt,
                "raw_prediction": result.raw_prediction,
            }
            write_json(subtype_dir / f"{example.id}.json", payload)
            (subtype_dir / f"{example.id}.txt").write_text(
                "\n".join(
                    [
                        f"ID: {example.id}",
                        f"Subtype: {example.diagnostic_subtype}",
                        f"Source mode: {example.source_mode}",
                        f"Gold answer: {result.gold_answer}",
                        f"Extracted prediction: {result.extracted_prediction}",
                        "",
                        "Prompt:",
                        example.prompt,
                        "",
                        "Model generation:",
                        result.raw_prediction,
                    ]
                ),
                encoding="utf-8",
            )
    return written_counts


def main(default_question_type: str | None = None) -> None:
    args = parse_args(default_question_type)
    paths = type_paths(args.question_type, data_dir=Path(args.data_dir))
    if not paths.train_csv.exists() or not paths.split_csv.exists():
        raise SystemExit(
            f"Missing diagnostic data for {args.question_type}. Run:\n"
            f"  python3 experiments/type_diagnostics/prepare_type_datasets.py --question-type {args.question_type}"
        )

    try:
        import kagglehub  # type: ignore
    except ImportError:
        kagglehub = None

    model_path = resolve_model_path(args, kagglehub)
    eval_examples = load_eval_examples(paths, args.eval_splits, args.max_eval_samples)
    if args.backend == "vllm":
        raw_predictions = generate_with_vllm(args, model_path, eval_examples)
    else:
        raw_predictions = generate_with_transformers(args, model_path, eval_examples)

    results = [
        score_prediction(
            example_id=example.id,
            category=example.category,
            raw_prediction=raw_prediction,
            gold_answer=example.answer,
        )
        for example, raw_prediction in zip(eval_examples, raw_predictions)
    ]

    total = len(results)
    correct = sum(1 for result in results if result.correct)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    by_subtype = summarize_by_subtype(eval_examples, results)
    failed_sample_counts = write_failed_trace_samples(
        report_dir,
        eval_examples,
        results,
        per_subtype=args.failed_traces_per_subtype,
    )

    summary = {
        "mode": "type_diagnostic_inference",
        "question_type": args.question_type,
        "category": QUESTION_TYPES[args.question_type]["category"],
        "total": total,
        "correct": correct,
        "accuracy": float(correct) / float(total) if total else 0.0,
        "by_subtype": by_subtype,
        "failed_trace_sample_counts": failed_sample_counts,
        "backend": args.backend,
        "model_path": model_path,
        "adapter_dir": str(Path(args.adapter_dir).resolve()),
        "train_csv": str(paths.train_csv.resolve()),
        "split_csv": str(paths.split_csv.resolve()),
        "eval_splits": args.eval_splits,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "metric": "reference/evaluation.py-compatible",
        "numeric_relative_tolerance": 1e-2,
        "numeric_absolute_tolerance": 1e-5,
    }
    write_json(report_dir / "metrics.json", summary)
    write_predictions(report_dir / "predictions.jsonl", eval_examples, results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
