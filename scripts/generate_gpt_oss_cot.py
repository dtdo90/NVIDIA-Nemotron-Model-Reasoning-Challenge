%%writefile generate_gpt_oss_cot.py
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from nemotron_baseline.data import infer_category
    from nemotron_baseline.metric import answers_match, extract_boxed_answer
    from nemotron_baseline.prompts import strip_wonderland_prefix
except ImportError:
    from data import infer_category  # type: ignore
    from metric import answers_match, extract_boxed_answer  # type: ignore
    from prompts import strip_wonderland_prefix  # type: ignore


LABEL_BY_CATEGORY = {
    "Bit Manipulation": "bitwise and binary transformation tasks",
    "Gravity": "physics-based quantitative reasoning tasks",
    "Unit Conversion": "unit and measurement transformation tasks",
    "Text Cipher": "textual cipher and string transformation tasks",
    "Transformation Rules": "symbolic and algebraic manipulation tasks",
    "Numeral System": "numerical representation conversion tasks",
}


DEBUG_TYPE_TO_CATEGORY = {
    "bit": "Bit Manipulation",
    "gravity": "Gravity",
    "conversion": "Unit Conversion",
    "cipher": "Text Cipher",
    "transformation": "Transformation Rules",
    "numeral": "Numeral System",
}


FINAL_ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:final answer|answer)\s*:.*$|^\s*the final answer is\s*.*$"
)
BOXED_LINE_RE = re.compile(r"(?i)\\boxed\{.*\}")
TRANSFORMATION_RULES_PROMPT_RE = re.compile(
    r"Below are a few examples:\n(?P<examples>.*?)\nNow, determine the result for: (?P<target>.*)",
    re.S,
)
STANDARD_NUMERIC_TRANSFORMATION_OPERATORS = {"+", "-", "*", "/"}


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


@dataclass(frozen=True)
class AttemptCandidate:
    attempt_index: int
    seed: int
    raw_text: str
    predicted_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT data locally with gpt-oss-120b served by vLLM."
    )
    parser.add_argument("--input-csv", default="/kaggle/input/datasets/taidoduc/nemotrone/train.csv")
    parser.add_argument("--output-csv", default="/kaggle/working/train_cot_gpt_oss.csv")
    parser.add_argument(
        "--failed-csv",
        default=None,
        help="Optional CSV path for failed rows. Defaults to <output stem>_failed.csv.",
    )
    parser.add_argument(
        "--model-path",
        default="/kaggle/input/gpt-oss-120b/transformers/default/1",
        help="Local path to the gpt-oss-120b model weights.",
    )
    parser.add_argument("--served-model-name", default="gpt-oss")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--client-host",
        default=None,
        help="Host used by the OpenAI client. Defaults to 127.0.0.1 when --host is 0.0.0.0.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="sk-local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--session-timeout", type=float, default=960.0)
    parser.add_argument("--server-timeout", type=int, default=180)
    parser.add_argument("--max-input-rows", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run only one sample per category type, chosen deterministically by sorted id.",
    )
    parser.add_argument(
        "--debug-type",
        choices=sorted(DEBUG_TYPE_TO_CATEGORY),
        default=None,
        help="When debugging, select only one sample from the chosen category alias.",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument(
        "--attempts",
        type=int,
        default=2,
        help="Number of parallel generations to run per row.",
    )
    parser.add_argument(
        "--attempt-workers",
        type=int,
        default=2,
        help="Maximum number of concurrent generation attempts per row.",
    )
    parser.add_argument(
        "--early-stop-votes",
        type=int,
        default=1,
        help="Stop collecting attempts once this many answers match the gold answer. Set 0 to disable.",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show a progress bar for pending rows.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable the progress bar.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Stream generated completion tokens to stderr. With multiple attempts, only attempt 1 is shown live.",
    )
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", dest="start_server", action="store_false")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--stream-interval", type=int, default=200)
    parser.add_argument("--preload-weights", action="store_true", default=True)
    parser.add_argument("--no-preload-weights", dest="preload_weights", action="store_false")
    parser.add_argument("--preload-workers", type=int, default=16)
    parser.add_argument("--vllm-log-file", default="vllm_gpt_oss_server.log")
    return parser.parse_args()


def require_runtime_dependencies():
    try:
        from openai import OpenAI  # type: ignore
        from openai_harmony import (  # type: ignore
            Conversation,
            HarmonyEncodingName,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
            ToolNamespaceConfig,
            load_harmony_encoding,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install `openai` and `openai_harmony`, plus vLLM."
        ) from exc

    return {
        "OpenAI": OpenAI,
        "Conversation": Conversation,
        "HarmonyEncodingName": HarmonyEncodingName,
        "Message": Message,
        "ReasoningEffort": ReasoningEffort,
        "Role": Role,
        "SystemContent": SystemContent,
        "ToolNamespaceConfig": ToolNamespaceConfig,
        "load_harmony_encoding": load_harmony_encoding,
    }


def load_input_rows(input_csv: str | Path, max_input_rows: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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


def select_debug_rows(
    rows: list[dict[str, str]],
    *,
    completed_ids: set[str] | None = None,
    debug_type: str | None = None,
) -> list[dict[str, str]]:
    completed = completed_ids or set()
    selected: list[dict[str, str]] = []
    seen_categories: set[str] = set()
    target_category = DEBUG_TYPE_TO_CATEGORY.get(debug_type) if debug_type else None

    for row in rows:
        if row["id"] in completed:
            continue
        category = infer_category(row["prompt"])
        if target_category is not None and category != target_category:
            continue
        if category in seen_categories:
            continue
        selected.append(row)
        seen_categories.add(category)
        if target_category is not None:
            break

    return selected


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def resolve_failed_output_path(output_path: Path, failed_csv: str | None) -> Path:
    if failed_csv:
        return Path(failed_csv)
    return output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}")


def label_for_prompt(prompt: str) -> tuple[str, str]:
    category = infer_category(prompt)
    return category, LABEL_BY_CATEGORY[category]


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

    return "\n".join(lines).strip()


def infer_transformation_rules_subtype(puzzle_prompt: str, gold_answer: str) -> str:
    match = TRANSFORMATION_RULES_PROMPT_RE.search(puzzle_prompt)
    if match is None:
        if any(char.isdigit() for char in f"{puzzle_prompt}{gold_answer}"):
            return "numeric_novel_operator"
        return "symbolic_string"

    example_block = match.group("examples")
    target_expression = match.group("target").strip()

    example_inputs: list[str] = []
    example_outputs: list[str] = []
    for line in example_block.splitlines():
        if " = " not in line:
            continue
        left_side, right_side = line.split(" = ", 1)
        example_inputs.append(left_side.strip())
        example_outputs.append(right_side.strip())

    combined_text = "".join(example_inputs) + "".join(example_outputs) + target_expression + gold_answer
    if not any(char.isdigit() for char in combined_text):
        return "symbolic_string"

    operator_characters: set[str] = set()
    for expression in [*example_inputs, target_expression]:
        for char in expression:
            if char.isdigit() or char.isspace():
                continue
            operator_characters.add(char)

    output_has_symbols = any(
        not char.isdigit() and not char.isspace()
        for char in "".join(example_outputs) + gold_answer
    )
    if (
        operator_characters
        and operator_characters.issubset(STANDARD_NUMERIC_TRANSFORMATION_OPERATORS)
        and not output_has_symbols
    ):
        return "numeric_standard"
    return "numeric_novel_operator"


def normalize_generated_cot(raw_text: str, gold_answer: str) -> str:
    reasoning = strip_final_answer_block(raw_text)
    final_line = f"Final Answer: The final answer is \\boxed{{{gold_answer}}}."
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


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        return None

    def set_postfix_str(self, _: str, refresh: bool = False) -> None:
        return None

    def write(self, message: str) -> None:
        print(message, file=sys.stderr)

    def clear(self) -> None:
        return None

    def refresh(self) -> None:
        return None

    def close(self) -> None:
        return None


def create_progress_bar(total: int, enabled: bool):
    if not enabled or total <= 0:
        return NullProgressBar()
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        print("tqdm is not installed; continuing without a progress bar.", file=sys.stderr)
        return NullProgressBar()
    return tqdm(total=total, desc="Generating CoT", unit="row", dynamic_ncols=True, file=sys.stderr)


def extract_stream_chunk_text(chunk) -> str:
    choices = getattr(chunk, "choices", None) or []
    text_parts: list[str] = []
    for choice in choices:
        text = getattr(choice, "text", None)
        if text:
            text_parts.append(text)
    return "".join(text_parts)


def preload_model_weights(model_path: str, workers: int) -> None:
    print(f"Loading model weights from {model_path} into OS page cache...", file=sys.stderr)
    start_time = time.time()

    files_to_load: list[str] = []
    total_size = 0
    for root, _, files in os.walk(model_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                files_to_load.append(file_path)
                total_size += os.path.getsize(file_path)

    def _read_file(path: str) -> None:
        with open(path, "rb") as file_object:
            while file_object.read(1024 * 1024 * 1024):
                pass

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(_read_file, files_to_load))

    elapsed = time.time() - start_time
    print(
        f"Preloaded {len(files_to_load)} files ({total_size / 1e9:.2f} GB) in {elapsed:.2f}s.",
        file=sys.stderr,
    )


class VLLMServerManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen[str] | None = None
        self.log_handle = None

    @property
    def base_url(self) -> str:
        client_host = self.args.client_host
        if client_host is None:
            client_host = "127.0.0.1" if self.args.host == "0.0.0.0" else self.args.host
        return f"http://{client_host}:{self.args.port}/v1"

    def start(self) -> None:
        if not self.args.start_server:
            return

        log_path = Path(self.args.vllm_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log_path.open("w", encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--seed",
            str(self.args.seed),
            "--model",
            self.args.model_path,
            "--served-model-name",
            self.args.served_model_name,
            "--tensor-parallel-size",
            str(self.args.tensor_parallel_size),
            "--max-num-seqs",
            str(self.args.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            "--dtype",
            self.args.dtype,
            "--kv-cache-dtype",
            self.args.kv_cache_dtype,
            "--max-model-len",
            str(self.args.max_model_len),
            "--stream-interval",
            str(self.args.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]

        env = os.environ.copy()
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        env.setdefault("TRANSFORMERS_NO_FLAX", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            text=True,
        )

    def wait_until_ready(self, client) -> None:
        start_time = time.time()
        for _ in range(self.args.server_timeout):
            if self.args.start_server and self.process is not None:
                return_code = self.process.poll()
                if return_code is not None:
                    if self.log_handle is not None:
                        self.log_handle.flush()
                    logs = Path(self.args.vllm_log_file).read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"vLLM server died with code {return_code}.\n{logs}")
            try:
                client.models.list()
                elapsed = time.time() - start_time
                print(f"vLLM server is ready in {elapsed:.2f}s.", file=sys.stderr)
                return
            except Exception:
                time.sleep(1)

        log_excerpt = ""
        log_path = Path(self.args.vllm_log_file)
        if log_path.exists():
            if self.log_handle is not None:
                self.log_handle.flush()
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-120:])
            if tail:
                log_excerpt = f"\nLast log lines from {log_path}:\n{tail}"

        raise RuntimeError(
            "vLLM server failed to become ready before timeout."
            f"{log_excerpt}"
        )

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self.log_handle is not None:
            self.log_handle.close()


class GPTOSSPromptBuilder:
    def __init__(self, deps: dict[str, object], reasoning_effort: str):
        self._conversation_cls = deps["Conversation"]
        self._message_cls = deps["Message"]
        self._role = deps["Role"]
        self._system_content_cls = deps["SystemContent"]
        self._tool_namespace_config = deps["ToolNamespaceConfig"]
        self._encoding = deps["load_harmony_encoding"](deps["HarmonyEncodingName"].HARMONY_GPT_OSS)
        self._reasoning_effort = getattr(deps["ReasoningEffort"], reasoning_effort.upper())
        self.stop_token_ids = self._encoding.stop_tokens_for_assistant_actions()

    @staticmethod
    def build_default_system_prompt() -> str:
        return (
            "You are a world-class logical reasoning competitor. "
            "You must solve each puzzle carefully and produce an exact final answer. "
            "You must place the final answer inside \\boxed{}."
        )

    @staticmethod
    def build_default_preference_prompt() -> str:
        return (
            "Reason step by step, keep the explanation concise but useful, and double-check "
            "transformations, arithmetic, symbols, and formatting before the final answer. "
            "Finish with exactly one line of the form: "
            "Final Answer: The final answer is \\boxed{...}."
        )

    @staticmethod
    def build_bit_manipulation_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for bit-manipulation puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example 8-bit input-output pairs\n"
            "2. the verified correct answer for the target input\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "transformation rule from the examples and applies it to the target input.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Reference example pairs when identifying the pattern.\n"
            "- Apply the inferred rule carefully to the target input.\n"
            "- Verify that the derived result matches the verified correct answer.\n"
            "- Double-check all bit transformations, binary strings, arithmetic, symbols, and formatting before the final answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- Preserve exact binary strings.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_bit_manipulation_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following bit-manipulation puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check all bit "
            "transformations, binary strings, arithmetic, symbols, and formatting before the final answer. "
            "Infer the hidden rule from the examples, citing the relevant example pairs as evidence. "
            "Then apply the rule carefully to the target input, show the intermediate bit-level reasoning "
            "where helpful, and confirm that the result matches the verified correct answer.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_unit_conversion_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for unit-conversion puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example input-output conversion pairs\n"
            "2. the verified correct answer for the target input\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "conversion rule from the examples and applies it to the target input.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Identify the conversion factor or formula suggested by the examples.\n"
            "- Reference example pairs when estimating and confirming the rule.\n"
            "- Apply the inferred rule carefully to the target input.\n"
            "- Verify that the derived result matches the verified correct answer.\n"
            "- Double-check arithmetic, rounding, units, symbols, decimal placement, and formatting before the final answer.\n"
            "- When decimals are involved, round the final converted value to 2 decimal places unless the verified correct answer uses a different exact format.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- Preserve exact numbers from the problem statement.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_unit_conversion_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following unit-conversion puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check arithmetic, "
            "rounding, units, symbols, decimal placement, and formatting before the final answer. "
            "First infer the conversion factor or formula from the examples by comparing several "
            "input-output pairs. Base the reasoning primarily on the example pairs; use the verified "
            "correct answer only as a final consistency check. Then apply the rule carefully to the "
            "target input. If the computed result is non-integer, round the final value to 2 decimal "
            "places unless the verified correct answer uses a different exact format.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_gravity_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for gravity-formula puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example time-distance observations\n"
            "2. the verified correct answer for the target input\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "gravitational constant from the examples and applies it to the target time using the stated formula.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Use the formula d = 0.5*g*t^2 as the basis of the reasoning.\n"
            "- Infer the hidden value of g from the example observations.\n"
            "- Reference example pairs when estimating and confirming g.\n"
            "- Apply the inferred value of g carefully to the target time.\n"
            "- Verify that the derived result matches the verified correct answer.\n"
            "- Double-check arithmetic, substitutions, units, symbols, decimal placement, and formatting before the final answer.\n"
            "- When decimals are involved, round the final value to 2 decimal places unless the verified correct answer uses a different exact format.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- Preserve exact numbers from the problem statement.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_gravity_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following gravity puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check arithmetic, "
            "substitutions, units, symbols, decimal placement, and formatting before the final answer. "
            "Use the formula d = 0.5*g*t^2 and infer the hidden gravitational constant from the examples "
            "by comparing several time-distance pairs. Base the reasoning primarily on the example pairs; "
            "use the verified correct answer only as a final consistency check. Then apply the inferred value "
            "of g carefully to the target time and verify that the result matches the verified correct answer. "
            "When decimals are involved, round the final value to 2 decimal places unless the verified correct "
            "answer uses a different exact format.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_text_cipher_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for text-cipher puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example encrypted-to-decrypted text pairs\n"
            "2. the verified correct answer for the target encrypted text\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "cipher rule from the examples and applies it to the target text.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Infer the letter-to-letter cipher mapping from the examples.\n"
            "- Reference example pairs when identifying the mapping.\n"
            "- Use repeated-letter patterns, short common words, and partially decoded words where helpful.\n"
            "- Apply the inferred mapping carefully to the target text.\n"
            "- Verify that the derived plaintext matches the verified correct answer.\n"
            "- Double-check letter mappings, decoded words, spelling, symbols, and formatting before the final answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- Preserve exact strings from the problem statement.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_text_cipher_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following text-cipher puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check letter "
            "mappings, decoded words, spelling, symbols, and formatting before the final answer. "
            "Use the examples to infer the hidden substitution rule, referencing example pairs when "
            "identifying the mapping. Use repeated-letter patterns, short words, and partially decoded "
            "words where helpful. Then apply the mapping carefully to the target text and verify that the "
            "result matches the verified correct answer.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_symbolic_transformation_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for symbol-transformation puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example symbol-string transformations\n"
            "2. the verified correct answer for the target string\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "symbol-level rewrite rule from the examples and applies it to the target string.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Work at the character level.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Compare input and output lengths, positions, repeated symbols, deleted symbols, substitutions, and reorderings.\n"
            "- Infer local rewrite patterns such as deletion, substitution, duplication, reordering, pairing, or collapsing repeated symbols.\n"
            "- Avoid speculative keyboard-layout or ASCII-code theories unless multiple examples clearly support them.\n"
            "- Preserve every symbol exactly, including quotes, slashes, backslashes, brackets, braces, and punctuation.\n"
            "- Apply the inferred rule carefully to the target string.\n"
            "- Verify that the derived string matches the verified correct answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_symbolic_transformation_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following symbol-transformation puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check every symbol, "
            "its position, and the final ordering before the final answer. Base the reasoning on the example "
            "pairs by comparing which characters are preserved, removed, substituted, duplicated, or reordered. "
            "Use the verified correct answer only as a final consistency check.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_numeric_standard_transformation_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for numeric transformation puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example numeric expressions and transformed results\n"
            "2. the verified correct answer for the target expression\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the arithmetic "
            "or digit-level rule from the examples and applies it to the target expression.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Infer the operator-specific arithmetic or digit-level transformation suggested by the examples.\n"
            "- Consider patterns such as sum, difference, product, concatenation, reversal, digit-wise operations, or simple operator-specific rewrites.\n"
            "- Reference multiple example pairs before committing to a rule.\n"
            "- Apply the inferred rule carefully to the target expression.\n"
            "- Verify that the derived result matches the verified correct answer.\n"
            "- Double-check arithmetic, digit order, signs, and formatting before the final answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_numeric_standard_transformation_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following numeric transformation puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check arithmetic, "
            "digit order, and formatting before the final answer. Use the examples to infer the operator-specific "
            "numeric rule, referencing several example pairs before applying it to the target expression. Use the "
            "verified correct answer only as a final consistency check.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_numeric_novel_operator_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for numeric puzzles with nonstandard operators or mixed-format outputs.\n\n"
            "You will be given:\n"
            "1. a puzzle with example transformed expressions\n"
            "2. the verified correct answer for the target expression\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the semantics "
            "of the nonstandard operators or output formatting from the examples and applies them to "
            "the target expression.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for the rule.\n"
            "- Treat each nonstandard operator or separator as meaningful and infer its role from repeated examples.\n"
            "- Consider arithmetic, digit-level transforms, concatenation, operator-specific formatting, and mixed numeric-symbolic outputs when supported by the examples.\n"
            "- Reference multiple example pairs before committing to a rule.\n"
            "- Preserve exact digits and symbols in the final result.\n"
            "- Apply the inferred rule carefully to the target expression.\n"
            "- Verify that the derived result matches the verified correct answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_numeric_novel_operator_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following numeric transformation puzzle with nonstandard operators.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check arithmetic, "
            "digit order, operator meaning, and formatting before the final answer. Use the examples to infer "
            "what each operator or separator does, referencing multiple example pairs before applying the rule "
            "to the target expression. Preserve exact digits and symbols in the final result. Use the verified "
            "correct answer only as a final consistency check.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    @staticmethod
    def build_numeral_system_system_prompt() -> str:
        return (
            "You are an expert logical reasoning solver generating high-quality step-by-step "
            "solutions for numeral-system conversion puzzles.\n\n"
            "You will be given:\n"
            "1. a puzzle with example number-to-numeral conversions\n"
            "2. the verified correct answer for the target number\n\n"
            "Your task is to produce a detailed but focused worked solution that infers the hidden "
            "numeral system from the examples and applies it to the target number.\n\n"
            "Requirements:\n"
            "- Reason step by step.\n"
            "- Keep the explanation concise but useful.\n"
            "- Use the examples as the main evidence for identifying the numeral system.\n"
            "- Reference example pairs when identifying the pattern.\n"
            "- Infer the numeral symbols and composition rules from the examples.\n"
            "- Decompose the target number into place values or standard numeral components as needed.\n"
            "- Apply the inferred numeral-system rules carefully to the target number.\n"
            "- Verify that the derived numeral matches the verified correct answer.\n"
            "- Double-check symbol choice, ordering, subtraction rules, arithmetic, and formatting before the final answer.\n"
            "- Do not provide multiple candidate rules or answers.\n"
            "- Preserve exact numbers and symbols from the problem statement.\n"
            "- End with exactly one line in this format:\n"
            "Final Answer: The final answer is \\boxed{<answer>}."
        )

    @staticmethod
    def build_numeral_system_user_prompt(puzzle_prompt: str, gold_answer: str) -> str:
        return (
            "Solve the following numeral-system conversion puzzle.\n\n"
            "Reason step by step, keep the explanation concise but useful, and double-check symbol "
            "choice, ordering, subtraction rules, arithmetic, and formatting before the final answer. "
            "Use the examples to infer the hidden numeral system, referencing example pairs when "
            "identifying the pattern. Then apply the numeral-system rules carefully to the target "
            "number and verify that the result matches the verified correct answer.\n\n"
            "### Problem ###\n"
            f"{puzzle_prompt}\n\n"
            "### Verified Correct Answer ###\n"
            f"{gold_answer}"
        )

    def build_prompts(self, category: str, puzzle_prompt: str, gold_answer: str) -> tuple[str, str]:
        cleaned_prompt = strip_wonderland_prefix(puzzle_prompt)
        if category == "Bit Manipulation":
            return (
                self.build_bit_manipulation_system_prompt(),
                self.build_bit_manipulation_user_prompt(cleaned_prompt, gold_answer),
            )
        if category == "Unit Conversion":
            return (
                self.build_unit_conversion_system_prompt(),
                self.build_unit_conversion_user_prompt(cleaned_prompt, gold_answer),
            )
        if category == "Gravity":
            return (
                self.build_gravity_system_prompt(),
                self.build_gravity_user_prompt(cleaned_prompt, gold_answer),
            )
        if category == "Text Cipher":
            return (
                self.build_text_cipher_system_prompt(),
                self.build_text_cipher_user_prompt(cleaned_prompt, gold_answer),
            )
        if category == "Transformation Rules":
            transformation_subtype = infer_transformation_rules_subtype(cleaned_prompt, gold_answer)
            if transformation_subtype == "symbolic_string":
                return (
                    self.build_symbolic_transformation_system_prompt(),
                    self.build_symbolic_transformation_user_prompt(cleaned_prompt, gold_answer),
                )
            if transformation_subtype == "numeric_standard":
                return (
                    self.build_numeric_standard_transformation_system_prompt(),
                    self.build_numeric_standard_transformation_user_prompt(cleaned_prompt, gold_answer),
                )
            return (
                self.build_numeric_novel_operator_system_prompt(),
                self.build_numeric_novel_operator_user_prompt(cleaned_prompt, gold_answer),
            )
        if category == "Numeral System":
            return (
                self.build_numeral_system_system_prompt(),
                self.build_numeral_system_user_prompt(cleaned_prompt, gold_answer),
            )

        system_prompt = self.build_default_system_prompt()
        preference_prompt = self.build_default_preference_prompt()
        user_prompt = f"{cleaned_prompt}\n\n{preference_prompt}"
        return system_prompt, user_prompt

    def build_prompt_token_ids(
        self,
        *,
        category: str,
        puzzle_prompt: str,
        gold_answer: str,
    ) -> list[int]:
        system_prompt, user_prompt = self.build_prompts(category, puzzle_prompt, gold_answer)
        tool_config = self._tool_namespace_config(
            name="python",
            description="No tools are available in this environment.",
            tools=[],
        )
        system_content = (
            self._system_content_cls.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=self._reasoning_effort)
            .with_tools(tool_config)
        )
        messages = [
            self._message_cls.from_role_and_content(self._role.SYSTEM, system_content),
            self._message_cls.from_role_and_content(self._role.USER, user_prompt),
        ]
        conversation = self._conversation_cls.from_messages(messages)
        return self._encoding.render_conversation_for_completion(conversation, self._role.ASSISTANT)


def generate_one(
    *,
    client,
    prompt_builder: GPTOSSPromptBuilder,
    served_model_name: str,
    row: dict[str, str],
    category: str,
    attempt_seed: int,
    args: argparse.Namespace,
    attempt: int,
    show_stream_output: bool,
    progress_bar,
) -> str:
    prompt_ids = prompt_builder.build_prompt_token_ids(
        category=category,
        puzzle_prompt=row["prompt"],
        gold_answer=row["answer"],
    )
    request_kwargs = dict(
        model=served_model_name,
        prompt=prompt_ids,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        seed=attempt_seed,
        timeout=args.session_timeout,
        extra_body={
            "min_p": args.min_p,
            "stop_token_ids": prompt_builder.stop_token_ids,
        },
    )
    stream = client.completions.create(stream=True, **request_kwargs)
    emitted_chunks: list[str] = []
    accumulated_text = ""
    if show_stream_output:
        progress_bar.clear()
        print(
            f"\n[stream] row={row['id']} category={category} attempt={attempt}",
            file=sys.stderr,
            flush=True,
        )
    try:
        for chunk in stream:
            chunk_text = extract_stream_chunk_text(chunk)
            if not chunk_text:
                continue
            emitted_chunks.append(chunk_text)
            accumulated_text += chunk_text
            if show_stream_output:
                print(chunk_text, end="", file=sys.stderr, flush=True)
            if "}" in chunk_text and extract_boxed_answer(accumulated_text):
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if show_stream_output:
        if emitted_chunks and not emitted_chunks[-1].endswith("\n"):
            print("", file=sys.stderr, flush=True)
        progress_bar.refresh()
    text = accumulated_text.strip()
    if not text:
        raise RuntimeError("gpt-oss returned an empty completion.")
    return text


def generate_attempt_batch(
    *,
    client,
    prompt_builder: GPTOSSPromptBuilder,
    served_model_name: str,
    row: dict[str, str],
    category: str,
    args: argparse.Namespace,
    retry_index: int,
    progress_bar,
) -> tuple[AttemptCandidate | None, list[str]]:
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1.")

    base_seed = args.seed + retry_index * max(1, args.attempts)
    errors: list[str] = []
    max_workers = min(args.attempt_workers, args.attempts)
    candidates: list[AttemptCandidate] = []
    first_candidate_with_answer: AttemptCandidate | None = None
    selected_candidate: AttemptCandidate | None = None
    correct_match_count = 0
    futures = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)

    try:
        for attempt_index in range(args.attempts):
            attempt_seed = (base_seed + attempt_index) ** 2
            future = executor.submit(
                generate_one,
                client=client,
                prompt_builder=prompt_builder,
                served_model_name=served_model_name,
                row=row,
                category=category,
                attempt_seed=attempt_seed,
                args=args,
                attempt=attempt_index + 1,
                show_stream_output=args.stream_output and attempt_index == 0,
                progress_bar=progress_bar,
            )
            futures[future] = (attempt_index + 1, attempt_seed)

        for future in as_completed(futures):
            attempt_number, attempt_seed = futures[future]
            try:
                raw_text = future.result()
                candidate = AttemptCandidate(
                    attempt_index=attempt_number,
                    seed=attempt_seed,
                    raw_text=raw_text,
                    predicted_answer=extract_final_answer(raw_text),
                )
                candidates.append(candidate)
                if candidate.predicted_answer and first_candidate_with_answer is None:
                    first_candidate_with_answer = candidate
                if candidate.predicted_answer and answers_match(candidate.predicted_answer, row["answer"]):
                    if selected_candidate is None:
                        selected_candidate = candidate
                    correct_match_count += 1
                    if args.early_stop_votes > 0 and correct_match_count >= args.early_stop_votes:
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"attempt {attempt_number} failed: {exc}")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if selected_candidate is not None:
        return selected_candidate, errors
    if first_candidate_with_answer is not None:
        return first_candidate_with_answer, errors
    if candidates:
        return candidates[0], errors
    return None, errors


def main() -> int:
    args = parse_args()
    if args.attempt_workers < 1:
        raise SystemExit("--attempt-workers must be at least 1.")
    if args.early_stop_votes < 0:
        raise SystemExit("--early-stop-votes must be non-negative.")
    if args.early_stop_votes > args.attempts:
        raise SystemExit("--early-stop-votes cannot exceed --attempts.")

    deps = require_runtime_dependencies()
    prompt_builder = GPTOSSPromptBuilder(deps, args.reasoning_effort)

    if args.preload_weights:
        preload_model_weights(args.model_path, args.preload_workers)

    server = VLLMServerManager(args)
    atexit.register(server.close)
    server.start()

    client = deps["OpenAI"](
        base_url=server.base_url,
        api_key=args.api_key,
        timeout=args.session_timeout,
    )
    server.wait_until_ready(client)
    input_rows = load_input_rows(args.input_csv, args.max_input_rows)
    output_path = Path(args.output_csv)
    failed_output_path = resolve_failed_output_path(output_path, args.failed_csv)
    ensure_parent_dir(output_path)
    ensure_parent_dir(failed_output_path)
    existing_success_ids = load_existing_ids(output_path) if args.resume else set()
    existing_failed_ids = load_existing_ids(failed_output_path) if args.resume else set()
    completed_ids = existing_success_ids | existing_failed_ids

    if args.debug or args.debug_type is not None:
        input_rows = select_debug_rows(
            input_rows,
            completed_ids=completed_ids,
            debug_type=args.debug_type,
        )
        debug_categories = [infer_category(row["prompt"]) for row in input_rows]
        print(
            f"Debug mode enabled: selected {len(input_rows)} rows "
            f"across categories {debug_categories}.",
            file=sys.stderr,
        )

    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    if (args.debug or args.debug_type is not None) and input_rows:
        preview_row = input_rows[0]
        preview_category = infer_category(preview_row["prompt"])
        preview_suffix = ""
        if preview_category == "Transformation Rules":
            preview_subtype = infer_transformation_rules_subtype(
                strip_wonderland_prefix(preview_row["prompt"]),
                preview_row["answer"],
            )
            preview_suffix = f"/{preview_subtype}"
        _, preview_user_prompt = prompt_builder.build_prompts(
            preview_category,
            preview_row["prompt"],
            preview_row["answer"],
        )
        print(
            (
                f"Debug user prompt preview for row {preview_row['id']} "
                f"({preview_category}{preview_suffix}):\n"
                "----- BEGIN USER PROMPT -----\n"
                f"{preview_user_prompt}\n"
                "----- END USER PROMPT -----"
            ),
            file=sys.stderr,
        )

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
    pending_rows = [row for row in input_rows if row["id"] not in completed_ids]
    pending_total = len(pending_rows)

    processed = 0
    written = len(existing_success_ids)
    skipped = len(input_rows) - pending_total
    failed = 0
    progress_bar = create_progress_bar(pending_total, enabled=args.progress)

    try:
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

            for index, row in enumerate(pending_rows, 1):
                category, label = label_for_prompt(row["prompt"])
                progress_bar.set_postfix_str(f"id={row['id']} category={category}", refresh=False)
                raw_text = ""
                verified = False
                last_error: Exception | None = None

                for attempt in range(1, args.max_retries + 1):
                    try:
                        selected_candidate, batch_errors = generate_attempt_batch(
                            client=client,
                            prompt_builder=prompt_builder,
                            served_model_name=args.served_model_name,
                            row=row,
                            category=category,
                            args=args,
                            retry_index=attempt - 1,
                            progress_bar=progress_bar,
                        )
                        if batch_errors:
                            last_error = RuntimeError("; ".join(batch_errors))
                        if selected_candidate is None:
                            raw_text = ""
                            if last_error is None:
                                last_error = RuntimeError("No attempt produced an extractable final answer.")
                            continue

                        raw_text = selected_candidate.raw_text
                        predicted_answer = selected_candidate.predicted_answer
                        verified = bool(predicted_answer) and answers_match(predicted_answer, row["answer"])
                        if verified or args.allow_unverified:
                            break
                        last_error = ValueError(
                            f"Predicted answer {predicted_answer!r} did not match gold answer "
                            f"{row['answer']!r}"
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
                            failure_reason=str(last_error) if last_error is not None else "no valid attempt",
                        ),
                    )
                    progress_bar.write(f"[{index}/{pending_total}] {row['id']} failed: {last_error}")
                    progress_bar.set_postfix_str(f"done={processed} failed={failed}", refresh=False)
                    progress_bar.update(1)
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
                    progress_bar.write(
                        f"[{index}/{pending_total}] {row['id']} skipped after retries: {last_error}",
                    )
                    progress_bar.set_postfix_str(f"done={processed} failed={failed}", refresh=False)
                    progress_bar.update(1)
                    continue

                output_row = GeneratedRow(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    generated_cot=normalize_generated_cot(raw_text, row["answer"]),
                    label=label,
                )
                write_row(writer, success_handle, output_row)
                processed += 1
                written += 1

                if args.sleep_between_requests > 0:
                    time.sleep(args.sleep_between_requests)

                if args.log_every > 0 and written % args.log_every == 0:
                    digest = stable_digest(row["prompt"], row["answer"])
                    progress_bar.write(
                        f"[{written}] wrote {row['id']} ({category}, {label}) digest={digest[:8]}",
                    )
                progress_bar.set_postfix_str(f"done={processed} failed={failed}", refresh=False)
                progress_bar.update(1)
    finally:
        progress_bar.close()

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "output_csv": str(output_path.resolve()),
        "failed_csv": str(failed_output_path.resolve()),
        "model_path": args.model_path,
        "served_model_name": args.served_model_name,
        "base_url": server.base_url,
        "start_server": args.start_server,
        "debug": args.debug,
        "debug_type": args.debug_type,
        "attempts": args.attempts,
        "attempt_workers": args.attempt_workers,
        "early_stop_votes": args.early_stop_votes,
        "progress": args.progress,
        "stream_output": args.stream_output,
        "seed": args.seed,
        "reasoning_effort": args.reasoning_effort,
        "resume": args.resume,
        "total_input_rows": len(input_rows),
        "pending_rows": pending_total,
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
    server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
