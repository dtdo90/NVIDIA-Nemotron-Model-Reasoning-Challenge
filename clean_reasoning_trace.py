#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nemotron_baseline.metric import answers_match, extract_boxed_answer
from nemotron_baseline.prompts import strip_wonderland_prefix


FINAL_ANSWER_LINE_RE = re.compile(r"(?im)^\s*\\boxed\{.*\}\s*$")
THINK_BLOCK_RE = re.compile(r"(?is)^\s*<think>\s*(.*?)\s*</think>\s*$")
FULL_REASONING_FORMAT_RE = re.compile(
    r"(?is)^\s*<think>\s*(.*?)\s*</think>\s*\\boxed\{.*\}\s*$"
)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class InputRow:
    id: str
    prompt: str
    answer: str
    raw_trace: str
    label: str


@dataclass(frozen=True)
class ApprovalResult:
    score: int
    decision: str
    strengths: list[str]
    issues: list[str]
    rationale: str


class NullProgressBar:
    def update(self, _: int = 1) -> None:
        return None

    def set_postfix_str(self, _: str, refresh: bool = False) -> None:
        return None

    def write(self, message: str) -> None:
        print(message, file=sys.stderr)

    def close(self) -> None:
        return None


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

        raise RuntimeError("vLLM server failed to become ready before timeout.")

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


class HarmonyPromptRenderer:
    def __init__(self, deps: dict[str, object], reasoning_effort: str):
        self._conversation_cls = deps["Conversation"]
        self._message_cls = deps["Message"]
        self._role = deps["Role"]
        self._system_content_cls = deps["SystemContent"]
        self._tool_namespace_config = deps["ToolNamespaceConfig"]
        self._encoding = deps["load_harmony_encoding"](deps["HarmonyEncodingName"].HARMONY_GPT_OSS)
        self._reasoning_effort = getattr(deps["ReasoningEffort"], reasoning_effort.upper())
        self.stop_token_ids = self._encoding.stop_tokens_for_assistant_actions()

    def render(self, system_prompt: str, user_prompt: str) -> list[int]:
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


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


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


def create_progress_bar(total: int, enabled: bool = True):
    if not enabled or total <= 0:
        return NullProgressBar()
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        print("tqdm is not installed; continuing without a progress bar.", file=sys.stderr)
        return NullProgressBar()
    return tqdm(total=total, desc="Cleaning CoT", unit="row", dynamic_ncols=True, file=sys.stderr)


def progress_write(progress_bar, message: str) -> None:
    progress_bar.write(message)


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
        if re.match(r"(?im)^\s*(?:final answer|answer)\s*:.*$", candidate) or "\\boxed{" in candidate:
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw reasoning traces and approve them for Nemotron SFT."
    )
    parser.add_argument("--input-csv", default="data/train_cot_gpt_oss.csv")
    parser.add_argument("--output-csv", default="data/train_cot_gpt_oss_clean.csv")
    parser.add_argument(
        "--failed-csv",
        default=None,
        help="Optional CSV path for failed rows. Defaults to <output stem>.failed.csv.",
    )
    parser.add_argument("--served-model-name", default="gpt-oss")
    parser.add_argument(
        "--approver-model-name",
        default=None,
        help="Optional served model name for the approver. Defaults to --served-model-name.",
    )
    parser.add_argument(
        "--model-path",
        default="/kaggle/input/gpt-oss-120b/transformers/default/1",
        help="Local path to the gpt-oss-120b model weights when starting a vLLM server.",
    )
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
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--cleaner-temperature", type=float, default=0.2)
    parser.add_argument("--approver-temperature", type=float, default=0.0)
    parser.add_argument("--cleaner-max-output-tokens", type=int, default=2048)
    parser.add_argument("--approver-max-output-tokens", type=int, default=1200)
    parser.add_argument("--score-threshold", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--start-server", action="store_true", default=True)
    parser.add_argument("--no-start-server", dest="start_server", action="store_false")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--stream-interval", type=int, default=200)
    parser.add_argument("--preload-weights", action="store_true", default=False)
    parser.add_argument("--no-preload-weights", dest="preload_weights", action="store_false")
    parser.add_argument("--preload-workers", type=int, default=16)
    parser.add_argument("--vllm-log-file", default="vllm_clean_reasoning_trace.log")
    return parser.parse_args()


def load_rows(input_csv: str | Path, max_input_rows: int | None) -> list[InputRow]:
    rows: list[InputRow] = []
    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            raw_trace = (row.get("generated_cot_raw") or row.get("generated_cot") or "").strip()
            rows.append(
                InputRow(
                    id=row["id"],
                    prompt=row["prompt"],
                    answer=row["answer"],
                    raw_trace=raw_trace,
                    label=row.get("label") or "",
                )
            )
            if max_input_rows is not None and len(rows) >= max_input_rows:
                break
    rows.sort(key=lambda row: row.id)
    return rows


def strip_verified_answer_block(problem: str) -> str:
    text = problem.strip()
    text = re.sub(
        r"(?is)\n+###\s*Verified Correct Answer\s*###\s*.*$",
        "",
        text,
    ).strip()
    return strip_wonderland_prefix(text)


def build_cleaner_system_prompt() -> str:
    return (
        "You are an expert editor for reasoning-trace datasets.\n\n"
        "Your task is to rewrite a raw solution into a clean, high-quality worked solution "
        "for supervised fine-tuning.\n\n"
        "Requirements:\n"
        "- Preserve the core reasoning steps that solve the problem.\n"
        "- Remove any early disclosure of the final answer.\n"
        "- Remove prompt references, meta-comments, self-talk, speculation, dead ends, and redundant scratch work.\n"
        "- Keep only the most coherent reasoning path.\n"
        "- Do not mention the existence of a verified answer, provided answer, label, ground truth, prompt instructions, or hidden metadata.\n"
        "- Do not introduce multiple candidate rules or answers.\n"
        "- Do not invent a completely different solution unless needed to repair an incomplete or obviously broken trace; prefer faithful compression and cleanup of the raw reasoning.\n"
        "- Repair obvious typos, malformed punctuation, mojibake, broken Unicode, and formatting corruption when the intended text is clear.\n"
        "- Remove leftover generation artifacts such as repeated instructions, scaffolding like 'Now produce final answer', channel markers, or tokens such as 'assistantfinal'.\n"
        "- Keep the reasoning natural, detailed enough for learning, and concise enough to avoid noise.\n"
        "- Preserve exact numbers, symbols, equations, units, binary strings, and formatting from the problem.\n"
        "- If a table or special formatting is corrupted, rewrite it as clean plain prose instead of preserving broken markup.\n"
        "- Use the gold answer only as a silent consistency check.\n"
        "- The final answer must appear only at the end.\n"
        "- Output must use exactly this structure:\n"
        "<think>\n"
        "...reasoning trace...\n"
        "</think>\n"
        "\\boxed{<answer>}\n"
        "- Do not put any extra text before <think> or after the final answer line.\n\n"
        "Output only the cleaned solution."
    )


def build_cleaner_user_prompt(problem: str, answer: str, raw_trace: str) -> str:
    return (
        "Rewrite the raw solution into a clean, high-quality worked solution for supervised fine-tuning.\n\n"
        "Important constraints:\n"
        "- Use the problem as the main source of reasoning.\n"
        "- Use the gold answer only as a silent consistency check.\n"
        "- Do not mention the gold answer, verified answer, provided answer, label, or ground truth.\n"
        "- Do not reveal the final answer before the final line.\n"
        "- Keep only one coherent reasoning path.\n"
        "- If the raw solution is incomplete or clearly broken, repair it minimally using the problem.\n"
        "- Fix obvious typos, encoding glitches, and leftover generation artifacts when the intended text is clear.\n"
        "- Preserve exact notation from the problem.\n"
        "- Put all reasoning inside a single <think>...</think> block.\n"
        "- After </think>, include exactly one final line in this form: \\boxed{<answer>}.\n"
        "- Do not put any text outside the <think> block except that final answer line.\n"
        "- Output only the cleaned solution.\n\n"
        "### Problem ###\n"
        f"{strip_verified_answer_block(problem)}\n\n"
        "### Gold Answer ###\n"
        f"{answer}\n\n"
        "### Raw Solution ###\n"
        f"{raw_trace}"
    )


def build_approver_system_prompt() -> str:
    return (
        "You are a strict quality approver for supervised fine-tuning reasoning traces.\n\n"
        "Your job is to evaluate a cleaned worked solution and decide whether it is suitable as a final training example.\n\n"
        "Score the solution on a 0-10 scale using the rubric below.\n\n"
        "Scoring rubric:\n"
        "- 10: Excellent. Clean, correct, natural, coherent, no answer leakage, no meta-talk, enough intermediate reasoning to teach the method, and final format is perfect.\n"
        "- 9: Very strong. Minor wording issues only, but clearly suitable for training.\n"
        "- 8: Good. Reasoning is correct and useful, with only small imperfections. Still suitable for training.\n"
        "- 7: Borderline. Mostly correct, but missing important reasoning, too compressed, slightly awkward, or has mild noise.\n"
        "- 6 or below: Not suitable. Includes substantial noise, weak reasoning, answer leakage, meta-commentary, formatting issues, incorrect or unsupported steps, or poor training value.\n\n"
        "Evaluate using these criteria:\n"
        "1. Correctness: the reasoning supports the final answer and does not contain clear mathematical or logical mistakes.\n"
        "2. Cleanliness: no prompt references, no mention of verified/provided answer, no hidden metadata, no self-talk, no raw scratchpad artifacts, no leftover generation instructions, and no obvious encoding corruption.\n"
        "3. No early answer leakage: the final answer is not revealed prematurely.\n"
        "4. Training value: the trace has enough intermediate reasoning to teach the method, but is not noisy, bloated, or garbled.\n"
        "5. Coherence: the solution follows one clear reasoning path without dead ends.\n"
        "6. Formatting: preserves exact symbols, avoids broken Unicode or mojibake, uses exactly one <think>...</think> block for the reasoning, and ends with exactly one final answer line in the required boxed format.\n\n"
        "Approval rule:\n"
        "- PASS only if score >= 8.\n"
        "- Otherwise return FAIL.\n\n"
        "Output rules:\n"
        "- Return valid JSON only.\n"
        "- Use exactly these keys:\n"
        '  "score": integer,\n'
        '  "decision": "PASS" or "FAIL",\n'
        '  "strengths": [list of short strings],\n'
        '  "issues": [list of short strings],\n'
        '  "rationale": "short paragraph"\n'
        "- Do not include any extra text outside the JSON."
    )


def build_approver_user_prompt(problem: str, answer: str, cleaned_solution: str) -> str:
    return (
        "Evaluate the following cleaned reasoning trace for supervised fine-tuning quality.\n\n"
        "### Problem ###\n"
        f"{strip_verified_answer_block(problem)}\n\n"
        "### Expected Final Answer ###\n"
        f"{answer}\n\n"
        "### Cleaned Solution ###\n"
        f"{cleaned_solution}"
    )


def call_completion(
    *,
    client,
    renderer: HarmonyPromptRenderer,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    min_p: float,
    timeout: float,
    seed: int,
) -> str:
    prompt_ids = renderer.render(system_prompt, user_prompt)
    response = client.completions.create(
        model=model_name,
        prompt=prompt_ids,
        temperature=temperature,
        max_tokens=max_output_tokens,
        seed=seed,
        timeout=timeout,
        extra_body={
            "min_p": min_p,
            "stop_token_ids": renderer.stop_token_ids,
        },
    )
    text = response.choices[0].text or ""
    text = text.strip()
    if not text:
        raise RuntimeError("Model returned an empty completion.")
    return text


def extract_json_payload(text: str) -> dict[str, object]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_RE.search(stripped)
    if not match:
        raise ValueError("No JSON object found in approver output.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Approver output JSON must be an object.")
    return payload


def coerce_approval(payload: dict[str, object]) -> ApprovalResult:
    score = int(payload["score"])
    decision = str(payload["decision"]).upper()
    strengths = [str(item) for item in payload.get("strengths", [])]
    issues = [str(item) for item in payload.get("issues", [])]
    rationale = str(payload.get("rationale", ""))
    if decision not in {"PASS", "FAIL"}:
        raise ValueError(f"Unexpected decision value: {decision!r}")
    return ApprovalResult(
        score=score,
        decision=decision,
        strengths=strengths,
        issues=issues,
        rationale=rationale,
    )


def normalize_cleaned_solution(cleaned_text: str, gold_answer: str) -> str:
    reasoning = strip_final_answer_block(cleaned_text)
    think_match = THINK_BLOCK_RE.match(reasoning)
    if think_match:
        think_content = think_match.group(1).strip()
    else:
        think_content = reasoning.strip()

    think_block = "<think>\n"
    if think_content:
        think_block += f"{think_content}\n"
    think_block += "</think>"
    final_line = f"\\boxed{{{gold_answer}}}"
    return f"{think_block}\n{final_line}"


def local_cleaning_issues(cleaned_text: str, gold_answer: str) -> list[str]:
    issues: list[str] = []
    extracted_answer = extract_final_answer(cleaned_text)
    if not extracted_answer or not answers_match(extracted_answer, gold_answer):
        issues.append("final answer does not match gold answer")

    if not FULL_REASONING_FORMAT_RE.match(cleaned_text.strip()):
        issues.append("missing required <think> block plus final answer format")
    if not FINAL_ANSWER_LINE_RE.search(cleaned_text):
        issues.append("missing exact boxed-answer final line format")

    reasoning_only = strip_final_answer_block(cleaned_text)
    think_match = THINK_BLOCK_RE.match(reasoning_only)
    if think_match is None:
        think_content = ""
    else:
        think_content = think_match.group(1).strip()
    if not think_content:
        issues.append("empty think block")

    banned_phrases = [
        "verified answer",
        "verified correct answer",
        "provided answer",
        "ground truth",
        "gold answer",
        "label",
    ]
    lowered = think_content.lower()
    for phrase in banned_phrases:
        if phrase in lowered:
            issues.append(f"contains banned phrase: {phrase}")
            break

    suspicious_artifacts = [
        ("assistantfinal", "contains channel artifact"),
        ("now produce final answer", "contains leftover generation instruction"),
        ("make sure to preserve formatting exactly", "contains leftover generation instruction"),
        ("‚ä", "contains mojibake"),
        ("‚ü", "contains mojibake"),
        ("�", "contains replacement-character corruption"),
    ]
    for needle, message in suspicious_artifacts:
        if needle in lowered:
            issues.append(message)
            break
    return issues


def write_output_row(writer: csv.DictWriter, handle, row: dict[str, object]) -> None:
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())


def resolve_failed_output_path(output_path: Path, failed_csv: str | None) -> Path:
    if failed_csv:
        return Path(failed_csv)
    return output_path.with_name(f"{output_path.stem}_failed{output_path.suffix}")


OUTPUT_FIELDNAMES = [
    "id",
    "prompt",
    "answer",
    "generated_cot",
    "label",
    "generated_cot_raw",
    "generated_cot_clean",
    "approval_score",
    "approval_decision",
    "approval_strengths",
    "approval_issues",
    "approval_rationale",
    "approval_passed",
]


def main() -> int:
    args = parse_args()
    approver_model_name = args.approver_model_name or args.served_model_name
    deps = require_runtime_dependencies()
    renderer = HarmonyPromptRenderer(deps, args.reasoning_effort)

    if args.preload_weights:
        preload_model_weights(args.model_path, args.preload_workers)

    server = VLLMServerManager(args)
    server.start()
    client = deps["OpenAI"](
        base_url=server.base_url,
        api_key=args.api_key,
        timeout=args.session_timeout,
    )
    server.wait_until_ready(client)

    rows = load_rows(args.input_csv, args.max_input_rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output_csv)
    failed_output_path = resolve_failed_output_path(output_path, args.failed_csv)
    ensure_parent_dir(output_path)
    ensure_parent_dir(failed_output_path)
    completed_ids = (
        load_processed_ids([output_path, failed_output_path]) if args.resume else set()
    )
    pending_rows = [row for row in rows if row.id not in completed_ids]
    pending_total = len(pending_rows)
    progress_bar = create_progress_bar(pending_total, enabled=True)

    success_mode = "a" if output_path.exists() and args.resume else "w"
    failed_mode = "a" if failed_output_path.exists() and args.resume else "w"
    write_success_header = (
        not output_path.exists()
        or success_mode == "w"
        or output_path.stat().st_size == 0
    )
    write_failed_header = (
        not failed_output_path.exists()
        or failed_mode == "w"
        or failed_output_path.stat().st_size == 0
    )

    processed = 0
    approved = 0
    failed = 0

    try:
        with (
            output_path.open(success_mode, newline="", encoding="utf-8") as success_handle,
            failed_output_path.open(failed_mode, newline="", encoding="utf-8") as failed_handle,
        ):
            success_writer = csv.DictWriter(
                success_handle,
                fieldnames=OUTPUT_FIELDNAMES,
                quoting=csv.QUOTE_MINIMAL,
            )
            failed_writer = csv.DictWriter(
                failed_handle,
                fieldnames=OUTPUT_FIELDNAMES,
                quoting=csv.QUOTE_MINIMAL,
            )
            if write_success_header:
                success_writer.writeheader()
                success_handle.flush()
                os.fsync(success_handle.fileno())
            if write_failed_header:
                failed_writer.writeheader()
                failed_handle.flush()
                os.fsync(failed_handle.fileno())

            for index, row in enumerate(pending_rows, 1):
                progress_bar.set_postfix_str(f"id={row.id}", refresh=False)
                cleaned_raw = ""
                normalized_clean = ""
                approval = ApprovalResult(0, "FAIL", [], [], "")
                local_issues: list[str] = []
                passed = False
                last_error: Exception | None = None

                try:
                    if not row.raw_trace:
                        local_issues = ["missing raw trace"]
                    else:
                        for attempt in range(1, args.max_retries + 1):
                            try:
                                cleaner_output = call_completion(
                                    client=client,
                                    renderer=renderer,
                                    model_name=args.served_model_name,
                                    system_prompt=build_cleaner_system_prompt(),
                                    user_prompt=build_cleaner_user_prompt(row.prompt, row.answer, row.raw_trace),
                                    temperature=args.cleaner_temperature,
                                    max_output_tokens=args.cleaner_max_output_tokens,
                                    min_p=args.min_p,
                                    timeout=args.session_timeout,
                                    seed=args.seed + index * 101 + attempt,
                                )
                                cleaned_raw = cleaner_output.strip()
                                local_issues = local_cleaning_issues(cleaned_raw, row.answer)
                                normalized_clean = (
                                    normalize_cleaned_solution(cleaned_raw, row.answer)
                                    if not local_issues
                                    else ""
                                )

                                approver_output = call_completion(
                                    client=client,
                                    renderer=renderer,
                                    model_name=approver_model_name,
                                    system_prompt=build_approver_system_prompt(),
                                    user_prompt=build_approver_user_prompt(
                                        row.prompt,
                                        row.answer,
                                        cleaned_raw,
                                    ),
                                    temperature=args.approver_temperature,
                                    max_output_tokens=args.approver_max_output_tokens,
                                    min_p=args.min_p,
                                    timeout=args.session_timeout,
                                    seed=args.seed + index * 103 + attempt,
                                )
                                approval = coerce_approval(extract_json_payload(approver_output))
                                passed = (
                                    approval.decision == "PASS"
                                    and approval.score >= args.score_threshold
                                    and not local_issues
                                )
                                if passed or attempt == args.max_retries:
                                    break
                            except Exception as exc:  # noqa: BLE001
                                last_error = exc
                                if attempt < args.max_retries:
                                    time.sleep(args.retry_delay * attempt)
                        if last_error is not None and not cleaned_raw and approval.score == 0:
                            local_issues = [str(last_error)]
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    local_issues = [str(exc)]

                output_row = {
                    "id": row.id,
                    "prompt": row.prompt,
                    "answer": row.answer,
                    "generated_cot": normalized_clean if passed else "",
                    "label": row.label,
                    "generated_cot_raw": row.raw_trace,
                    "generated_cot_clean": normalized_clean if normalized_clean else cleaned_raw,
                    "approval_score": approval.score,
                    "approval_decision": approval.decision,
                    "approval_strengths": json.dumps(approval.strengths, ensure_ascii=False),
                    "approval_issues": json.dumps(approval.issues + local_issues, ensure_ascii=False),
                    "approval_rationale": approval.rationale,
                    "approval_passed": "true" if passed else "false",
                }

                processed += 1
                if passed:
                    write_output_row(success_writer, success_handle, output_row)
                    approved += 1
                else:
                    write_output_row(failed_writer, failed_handle, output_row)
                    failed += 1

                if args.log_every > 0 and processed % args.log_every == 0:
                    progress_write(
                        progress_bar,
                        f"[{processed}] cleaned {row.id} pass={passed} score={approval.score}",
                    )
                progress_bar.set_postfix_str(
                    f"done={processed} approved={approved} failed={failed}",
                    refresh=False,
                )
                progress_bar.update(1)
    finally:
        progress_bar.close()
        server.close()

    summary = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "output_csv": str(output_path.resolve()),
        "failed_csv": str(failed_output_path.resolve()),
        "served_model_name": args.served_model_name,
        "approver_model_name": approver_model_name,
        "score_threshold": args.score_threshold,
        "resume": args.resume,
        "total_input_rows": len(rows),
        "pending_rows": pending_total,
        "processed_this_run": processed,
        "approved_this_run": approved,
        "failed_this_run": failed,
        "skipped_existing": len(rows) - pending_total,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
