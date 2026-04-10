#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.data import infer_category
from nemotron_baseline.metric import answers_match, extract_boxed_answer


LABEL_BY_CATEGORY = {
    "Bit Manipulation": "bitwise and binary transformation tasks",
    "Gravity": "physics-based quantitative reasoning tasks",
    "Unit Conversion": "unit and measurement transformation tasks",
    "Text Cipher": "textual cipher and string transformation tasks",
    "Transformation Rules": "symbolic and algebraic manipulation tasks",
    "Numeral System": "numerical representation conversion tasks",
}


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final answer|answer)\s*:.*$|^\s*the final answer is\s*.*$"
)

BOXED_LINE_RE = re.compile(r"(?i)\\boxed\{.*\}")


@dataclass(frozen=True)
class GeneratedRow:
    id: str
    prompt: str
    answer: str
    generated_cot: str
    label: str


@dataclass(frozen=True)
class FailedRow:
    id: str
    prompt: str
    answer: str
    generated_cot_raw: str
    label: str
    failure_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT data with Gemini for Nemotron reasoning training."
    )
    parser.add_argument("--input-csv", default="data/train.csv")
    parser.add_argument("--output-csv", default="data/train_cot_gemini.csv")
    parser.add_argument(
        "--failed-csv",
        default=None,
        help="Optional CSV path for failed rows. Defaults to <output stem>.failed.csv.",
    )
    parser.add_argument("--model", default="gemini-3-flash")
    parser.add_argument(
        "--api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable containing the Gemini API key.",
    )
    parser.add_argument(
        "--api-version",
        default="v1beta",
        help="Google Generative Language API version to use.",
    )
    parser.add_argument("--max-input-rows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="If set, keep rows even when Gemini's extracted final answer does not match the gold answer.",
    )
    return parser.parse_args()


def load_input_rows(input_csv: str | Path, max_input_rows: int | None) -> list[dict[str, str]]:
    rows = []
    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]})
            if max_input_rows is not None and len(rows) >= max_input_rows:
                break
    rows.sort(key=lambda row: row["id"])
    return rows


def load_existing_ids(output_csv: str | Path) -> set[str]:
    path = Path(output_csv)
    if not path.exists() or path.stat().st_size == 0:
        return set()

    completed: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = row.get("id")
            if row_id:
                completed.add(row_id)
    return completed


def load_processed_ids(paths: list[str | Path]) -> set[str]:
    processed: set[str] = set()
    for path in paths:
        processed.update(load_existing_ids(path))
    return processed


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def resolve_failed_output_path(output_path: Path, failed_csv: str | None) -> Path:
    if failed_csv:
        return Path(failed_csv)
    return output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}")


def label_for_prompt(prompt: str) -> tuple[str, str]:
    category = infer_category(prompt)
    return category, LABEL_BY_CATEGORY[category]


def build_prompt(prompt: str) -> tuple[str, str]:
    system = (
        "You are writing high-quality chain-of-thought examples for a reasoning benchmark. "
        "Solve the puzzle carefully, but keep the reasoning concise and useful. "
        "End with exactly one final-answer line in the form: "
        "Final Answer: The final answer is $\\boxed{...}$. "
        "Do not include multiple candidate answers, and do not repeat the final answer elsewhere."
    )
    user = (
        f"{prompt}\n\n"
        "Write a concise derivation or explanation that shows the key steps, then finish with the final-answer line."
    )
    return system, user


def _candidate_text_from_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        text = part.get("text")
        if text:
            texts.append(text)
    return "".join(texts).strip()


def _request_payload(system: str, user: str, *, temperature: float, top_p: float, max_output_tokens: int) -> dict[str, Any]:
    return {
        "systemInstruction": {
            "parts": [{"text": system}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens,
        },
    }


def call_gemini_api(
    *,
    api_key: str,
    api_version: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str:
    base = f"https://generativelanguage.googleapis.com/{api_version}/models/{urllib.parse.quote(model, safe='')}:generateContent"
    url = f"{base}?key={urllib.parse.quote(api_key)}"
    data = json.dumps(
        _request_payload(
            system,
            user,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = _candidate_text_from_response(payload)
    if not text:
        raise RuntimeError(f"Gemini response did not contain candidate text: {payload}")
    return text


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return ""


def strip_final_answer_block(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()

    while lines:
        candidate = lines[-1].strip()
        if not candidate:
            lines.pop()
            continue
        if FINAL_ANSWER_LINE_RE.match(candidate) or BOXED_LINE_RE.search(candidate):
            lines.pop()
            continue
        break

    cleaned = "\n".join(lines).strip()
    return cleaned


def normalize_generated_cot(raw_text: str, gold_answer: str) -> str:
    reasoning = strip_final_answer_block(raw_text)
    final_line = f"Final Answer: The final answer is $\\boxed{{{gold_answer}}}$"
    if reasoning:
        return f"{reasoning}\n\n{final_line}"
    return final_line


def stable_digest(prompt: str, answer: str) -> str:
    payload = f"{prompt}\n{answer}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def write_row(writer: csv.DictWriter, handle, row: GeneratedRow) -> None:
    writer.writerow(
        {
            "id": row.id,
            "prompt": row.prompt,
            "answer": row.answer,
            "generated_cot": row.generated_cot,
            "label": row.label,
        }
    )
    handle.flush()
    os.fsync(handle.fileno())


def write_failed_row(writer: csv.DictWriter, handle, row: FailedRow) -> None:
    writer.writerow(
        {
            "id": row.id,
            "prompt": row.prompt,
            "answer": row.answer,
            "generated_cot_raw": row.generated_cot_raw,
            "label": row.label,
            "failure_reason": row.failure_reason,
        }
    )
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set the {args.api_key_env} environment variable before running."
        )

    input_rows = load_input_rows(args.input_csv, args.max_input_rows)
    output_path = Path(args.output_csv)
    failed_output_path = resolve_failed_output_path(output_path, args.failed_csv)
    ensure_parent_dir(output_path)
    ensure_parent_dir(failed_output_path)
    existing_success_ids = load_existing_ids(output_path) if args.resume else set()
    existing_failed_ids = load_existing_ids(failed_output_path) if args.resume else set()
    completed_ids = existing_success_ids | existing_failed_ids

    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    success_mode = "a" if output_path.exists() and args.resume else "w"
    failed_mode = "a" if failed_output_path.exists() and args.resume else "w"
    write_success_header = (
        not output_path.exists() or success_mode == "w" or output_path.stat().st_size == 0
    )
    write_failed_header = (
        not failed_output_path.exists()
        or failed_mode == "w"
        or failed_output_path.stat().st_size == 0
    )

    processed = 0
    written = len(existing_success_ids)
    skipped = 0
    failed = 0

    with (
        output_path.open(success_mode, newline="", encoding="utf-8") as success_handle,
        failed_output_path.open(failed_mode, newline="", encoding="utf-8") as failed_handle,
    ):
        writer = csv.DictWriter(
            success_handle,
            fieldnames=["id", "prompt", "answer", "generated_cot", "label"],
            quoting=csv.QUOTE_MINIMAL,
        )
        failed_writer = csv.DictWriter(
            failed_handle,
            fieldnames=["id", "prompt", "answer", "generated_cot_raw", "label", "failure_reason"],
            quoting=csv.QUOTE_MINIMAL,
        )
        if write_success_header:
            writer.writeheader()
            success_handle.flush()
            os.fsync(success_handle.fileno())
        if write_failed_header:
            failed_writer.writeheader()
            failed_handle.flush()
            os.fsync(failed_handle.fileno())

        for index, row in enumerate(input_rows, 1):
            if row["id"] in completed_ids:
                skipped += 1
                continue

            category, label = label_for_prompt(row["prompt"])
            system, user = build_prompt(row["prompt"])

            raw_text = ""
            verified = False
            last_error: Exception | None = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    raw_text = call_gemini_api(
                        api_key=api_key,
                        api_version=args.api_version,
                        model=args.model,
                        system=system,
                        user=user,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_output_tokens=args.max_output_tokens,
                    )
                    predicted_answer = extract_final_answer(raw_text)
                    verified = bool(predicted_answer) and answers_match(predicted_answer, row["answer"])
                    if verified or args.allow_unverified:
                        break
                    last_error = ValueError(
                        f"Extracted answer {predicted_answer!r} did not match gold answer {row['answer']!r}"
                    )
                except Exception as exc:  # noqa: BLE001
                    last_error = exc

                if attempt < args.max_retries:
                    time.sleep(args.retry_delay * attempt)

            if not raw_text:
                failed += 1
                write_failed_row(
                    failed_writer,
                    failed_handle,
                    FailedRow(
                        id=row["id"],
                        prompt=row["prompt"],
                        answer=row["answer"],
                        generated_cot_raw="",
                        label=label,
                        failure_reason=str(last_error) if last_error is not None else "empty completion",
                    ),
                )
                print(f"[{index}/{len(input_rows)}] {row['id']} failed: {last_error}", file=sys.stderr)
                continue

            if not verified and not args.allow_unverified:
                failed += 1
                write_failed_row(
                    failed_writer,
                    failed_handle,
                    FailedRow(
                        id=row["id"],
                        prompt=row["prompt"],
                        answer=row["answer"],
                        generated_cot_raw=raw_text,
                        label=label,
                        failure_reason=str(last_error) if last_error is not None else "answer mismatch",
                    ),
                )
                print(
                    f"[{index}/{len(input_rows)}] {row['id']} skipped after retries: {last_error}",
                    file=sys.stderr,
                )
                continue

            generated_cot = normalize_generated_cot(raw_text, row["answer"])
            output_row = GeneratedRow(
                id=row["id"],
                prompt=row["prompt"],
                answer=row["answer"],
                generated_cot=generated_cot,
                label=label,
            )
            write_row(writer, success_handle, output_row)
            processed += 1
            written += 1

            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

            if args.log_every > 0 and written % args.log_every == 0:
                digest = stable_digest(row["prompt"], row["answer"])
                print(
                    f"[{written}] wrote {row['id']} ({category}, {label}) digest={digest[:8]}",
                    file=sys.stderr,
                )

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "output_csv": str(output_path.resolve()),
        "failed_csv": str(failed_output_path.resolve()),
        "model": args.model,
        "api_version": args.api_version,
        "resume": args.resume,
        "total_input_rows": len(input_rows),
        "already_completed": len(completed_ids),
        "already_succeeded": len(existing_success_ids),
        "already_failed": len(existing_failed_ids),
        "processed_this_run": processed,
        "written_total": written,
        "skipped_existing": skipped,
        "failed": failed,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
