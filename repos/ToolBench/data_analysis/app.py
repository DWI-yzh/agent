#!/usr/bin/env python3
"""Local web app for ToolBench stage 2 trajectory analysis and SAO conversion."""

from __future__ import annotations

import argparse
import json
import re
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DATA_ROOT = ROOT / "data_samples"
ANNOTATION_PATH = ROOT / "annotations" / "records.json"

ACTION_RE = re.compile(r"(?mi)^\s*Action\s*:\s*([^\r\n]+)\s*$")
ACTION_INPUT_RE = re.compile(r"(?mis)^\s*Action Input\s*:\s*(.*)$")
THOUGHT_RE = re.compile(r"(?mis)^\s*Thought\s*:\s*(.*?)(?=^\s*Action\s*:)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def load_samples() -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for group in ("g1", "g2", "g3"):
        path = DATA_ROOT / f"{group}_10.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing sample file: {path}")
        for record in load_jsonl(path):
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Sample in {path} has no valid sample_id")
            if sample_id in samples:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            samples[sample_id] = record
    return samples


def parse_jsonish(raw: str) -> tuple[Any, bool]:
    text = raw.strip()
    if not text:
        return {}, True
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return text, False


def parse_action(message: str) -> dict[str, Any] | None:
    action_matches = ACTION_RE.findall(message)
    if not action_matches:
        return None
    # Some generated messages contain a prose-like ``Action: Finish->...`` line
    # followed by the actual canonical ``Action: Finish``.  The last Action line
    # is the one paired with Action Input and is therefore the executable action.
    tool_name = action_matches[-1].strip()
    input_match = ACTION_INPUT_RE.search(message)
    arguments_raw = input_match.group(1).strip() if input_match else "{}"
    arguments, arguments_valid = parse_jsonish(arguments_raw)
    thought_match = THOUGHT_RE.search(message)
    thought = thought_match.group(1).strip() if thought_match else ""
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "arguments_raw": arguments_raw,
        "arguments_valid_json": arguments_valid,
        "thought": thought,
    }


def classify_observation(raw: str, tool_name: str) -> str:
    if tool_name == "Finish":
        return "success"
    lowered = raw.lower()
    error_markers = (
        "timeout",
        "timed out",
        "not exist",
        "does not exist",
        "unrecognized",
        "invalid",
        "must be",
        "error",
        "failed",
        "not found",
        "unauthorized",
        "forbidden",
    )
    if any(marker in lowered for marker in error_markers):
        return "error"
    compact = re.sub(r"\s+", "", raw)
    if compact in {"", "[]", "{}", '""'} or '"response":"[]"' in compact:
        return "empty"
    return "success"


def finish_observation(action: dict[str, Any]) -> str:
    arguments = action.get("arguments")
    return_type = arguments.get("return_type") if isinstance(arguments, dict) else None
    if return_type == "give_answer":
        return json.dumps(
            {"response": "successfully giving the final answer."}, ensure_ascii=False
        )
    return json.dumps({"response": "chose to give up and restart"}, ensure_ascii=False)


def derive_failure_type(
    action: dict[str, Any], observation_status: str, observation_raw: str
) -> str | None:
    if action["tool_name"] == "Finish":
        arguments = action.get("arguments")
        if isinstance(arguments, dict) and arguments.get("return_type") == "give_up_and_restart":
            return "premature_stop"
        return None
    if observation_status != "error":
        return None
    lowered = observation_raw.lower()
    if "must be" in lowered or "integer" in lowered or "type" in lowered:
        return "argument_type_error"
    if "unrecognized" in lowered or "invalid" in lowered:
        return "argument_value_error"
    if "endpoint" in lowered or "does not exist" in lowered:
        return "tool_selection_error"
    if "timeout" in lowered or "timed out" in lowered:
        return "environment_error"
    return "unknown_error"


def parse_sample(sample: dict[str, Any]) -> dict[str, Any]:
    conversations = sample.get("conversations", [])
    steps: list[dict[str, Any]] = []
    pending_control_messages: list[str] = []
    initial_user_seen = False

    for index, message in enumerate(conversations):
        role = message.get("from", "")
        value = message.get("value", "")
        if role == "user":
            if not initial_user_seen:
                initial_user_seen = True
            else:
                pending_control_messages.append(value)
            continue
        if role != "assistant":
            continue
        action = parse_action(value)
        if action is None:
            continue

        observation_raw = ""
        observation_role = "synthetic"
        for later in conversations[index + 1 :]:
            later_role = later.get("from", "")
            if later_role == "function":
                observation_raw = later.get("value", "")
                observation_role = "function"
                break
            if later_role == "assistant" and parse_action(later.get("value", "")):
                break
        if not observation_raw and action["tool_name"] == "Finish":
            observation_raw = finish_observation(action)

        observation_value, observation_valid_json = parse_jsonish(observation_raw)
        status = classify_observation(observation_raw, action["tool_name"])
        failure_type = derive_failure_type(action, status, observation_raw)
        is_final = action["tool_name"] == "Finish"
        action_valid = status == "success" and not (
            is_final
            and isinstance(action.get("arguments"), dict)
            and action["arguments"].get("return_type") == "give_up_and_restart"
        )

        steps.append(
            {
                "step_id": len(steps) + 1,
                "message_index": index,
                "thought": action["thought"],
                "action": {
                    "tool_name": action["tool_name"],
                    "arguments": action["arguments"],
                    "arguments_raw": action["arguments_raw"],
                    "arguments_valid_json": action["arguments_valid_json"],
                },
                "observation": {
                    "status": status,
                    "result": observation_value,
                    "raw": observation_raw,
                    "valid_json": observation_valid_json,
                    "source_role": observation_role,
                },
                "control_messages": pending_control_messages,
                "derived_labels": {
                    "is_final": is_final,
                    "action_valid": action_valid,
                    "recoverable": status == "error" and not is_final,
                    "failure_type": failure_type,
                    "training_use": (
                        "sft_positive"
                        if action_valid
                        else "dpo_rejected_candidate"
                        if status == "error"
                        else "error_analysis_only"
                    ),
                },
            }
        )
        pending_control_messages = []

    for step_index, step in enumerate(steps):
        history = [
            {
                "step_id": previous["step_id"],
                "action": {
                    "tool_name": previous["action"]["tool_name"],
                    "arguments": previous["action"]["arguments"],
                },
                "observation": {
                    "status": previous["observation"]["status"],
                    "result": previous["observation"]["result"],
                },
            }
            for previous in steps[:step_index]
        ]
        control_feedback = [
            feedback
            for previous in steps[: step_index + 1]
            for feedback in previous.get("control_messages", [])
        ]
        last_observation = history[-1]["observation"] if history else None
        step["state"] = {
            "user_query": sample.get("query", ""),
            "available_tools": sample.get("original_tools", []),
            "history": history,
            "history_steps": len(history),
            "last_observation": last_observation,
            "control_feedback": control_feedback,
        }

    return {
        "sample_id": sample["sample_id"],
        "sample_uid": sample.get("sample_uid"),
        "group": sample.get("group"),
        "query": sample.get("query", ""),
        "outcome": sample.get("outcome"),
        "original_tools": sample.get("original_tools", []),
        "tool_count": sample.get("tool_count", 0),
        "action_count": len(steps),
        "classification_basis": sample.get("classification_basis"),
        "steps": steps,
        "conversations": conversations,
    }


class AnnotationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}\n", encoding="utf-8")

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Cannot read annotation store {self.path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Annotation store root must be an object")
        return value

    def read_all(self) -> dict[str, Any]:
        with self.lock:
            return self._read_unlocked()

    def read_sample(self, sample_id: str) -> dict[str, Any]:
        return self.read_all().get(sample_id, {})

    def update(self, sample_id: str, section: str, payload: dict[str, Any]) -> None:
        if section not in {"trajectory", "sao"}:
            raise ValueError("Invalid annotation section")
        with self.lock:
            records = self._read_unlocked()
            sample_record = records.setdefault(sample_id, {})
            sample_record[section] = payload
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)


SAMPLES = load_samples()
PARSED_SAMPLES = {sample_id: parse_sample(sample) for sample_id, sample in SAMPLES.items()}
STORE = AnnotationStore(ANNOTATION_PATH)


def summary_for(sample: dict[str, Any], annotations: dict[str, Any]) -> dict[str, Any]:
    trajectory = annotations.get("trajectory", {})
    sao = annotations.get("sao", {})
    trajectory_steps = trajectory.get("steps", {}) if isinstance(trajectory, dict) else {}
    sao_steps = sao.get("steps", {}) if isinstance(sao, dict) else {}
    return {
        "sample_id": sample["sample_id"],
        "group": sample["group"],
        "query": sample["query"],
        "outcome": sample["outcome"],
        "action_count": sample["action_count"],
        "trajectory_completed_steps": sum(
            bool(step.get("complete")) for step in trajectory_steps.values()
        ),
        "trajectory_complete": bool(trajectory.get("complete")),
        "sao_completed_steps": sum(
            bool(step.get("confirmed")) for step in sao_steps.values()
        ),
        "sao_complete": bool(sao.get("complete")),
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ToolBenchDataAnalysis/1.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: HTTPStatus) -> None:
        self.send_json({"error": message}, status)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 2_000_000:
            raise ValueError("Request body must be between 1 byte and 2 MB")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path == "/api/samples":
            annotations = STORE.read_all()
            summaries = [
                summary_for(PARSED_SAMPLES[sample_id], annotations.get(sample_id, {}))
                for sample_id in sorted(PARSED_SAMPLES)
            ]
            self.send_json({"samples": summaries})
            return
        if path.startswith("/api/samples/"):
            sample_id = urllib.parse.unquote(path.removeprefix("/api/samples/"))
            sample = PARSED_SAMPLES.get(sample_id)
            if sample is None:
                self.send_error_json("Sample not found", HTTPStatus.NOT_FOUND)
                return
            self.send_json({"sample": sample, "annotations": STORE.read_sample(sample_id)})
            return
        if path == "/api/export/trajectory":
            self.send_export("trajectory", "toolbench_trajectory_annotations.jsonl")
            return
        if path == "/api/export/sao":
            self.send_export("sao", "state_action_observation_examples.jsonl")
            return
        self.send_static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed_url = urllib.parse.urlparse(self.path)
        match = re.fullmatch(r"/api/annotations/([^/]+)/(trajectory|sao)", parsed_url.path)
        if not match:
            self.send_error_json("Endpoint not found", HTTPStatus.NOT_FOUND)
            return
        sample_id = urllib.parse.unquote(match.group(1))
        section = match.group(2)
        if sample_id not in PARSED_SAMPLES:
            self.send_error_json("Sample not found", HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json_body()
            STORE.update(sample_id, section, payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"ok": True, "sample_id": sample_id, "section": section})

    def send_export(self, section: str, filename: str) -> None:
        records = STORE.read_all()
        lines: list[str] = []
        if section == "trajectory":
            for sample_id in sorted(records):
                trajectory = records[sample_id].get("trajectory")
                if trajectory:
                    lines.append(
                        json.dumps(
                            {"sample_id": sample_id, **trajectory}, ensure_ascii=False
                        )
                    )
        else:
            for sample_id in sorted(records):
                sao = records[sample_id].get("sao", {})
                steps = sao.get("steps", {}) if isinstance(sao, dict) else {}
                for step_id in sorted(steps, key=lambda value: int(value)):
                    lines.append(json.dumps(steps[step_id], ensure_ascii=False))
        body = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error_json("Invalid path", HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file():
            target = WEB_ROOT / "index.html"
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
        }
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_types.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not WEB_ROOT.is_dir():
        raise FileNotFoundError(f"Missing web directory: {WEB_ROOT}")
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"ToolBench data analysis app: {url}")
    print("Press Ctrl+C to stop.")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
