#!/usr/bin/env python3
"""Deterministically sample complete G1/G2/G3 ToolBench trajectories.

The released repository only contains five examples per group.  This script uses
the validation split of a ToolBench SFT mirror and cross-references the original
G2/G3 query indexes.  G1 is identified conservatively: the query is absent from
the G2/G3 indexes and the system prompt exposes exactly one original tool.

Only records whose final assistant message calls ``Finish`` are eligible.  This
avoids sampling the intermediate prefixes produced by ToolBench preprocessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


GROUPS = ("G1", "G2", "G3")
ACTION_RE = re.compile(r"(?m)^Action:\s*([^\r\n]+)\s*$")
FINISH_RE = re.compile(r"(?m)^Action:\s*Finish\s*$")
ORIGINAL_TOOL_RE = re.compile(r"(?m)^\d+\.([^:\r\n]+):")
RETURN_TYPE_RE = re.compile(r'"return_type"\s*:\s*"([^"]+)"')
STEP_RE = re.compile(r"^Step\s+(\d+):")
SOURCE_URLS = {
    "trajectory_parquet": (
        "https://huggingface.co/datasets/tuandunghcmut/toolbench-v1/resolve/"
        "main/data/validation-00000-of-00001.parquet?download=true"
    ),
    "g2_query_index": (
        "https://huggingface.co/datasets/Tool-COLT/ToolBenchG2/resolve/"
        "main/queries.jsonl?download=true"
    ),
    "g3_query_index": (
        "https://huggingface.co/datasets/Tool-COLT/ToolBenchG3/resolve/"
        "main/queries.jsonl?download=true"
    ),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    analysis_dir = script_dir.parent
    cache_dir = analysis_dir / "source_cache"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-parquet",
        type=Path,
        default=cache_dir / "validation-00000-of-00001.parquet",
        help="ToolBench SFT parquet containing id and conversations columns.",
    )
    parser.add_argument(
        "--g2-query-index",
        type=Path,
        default=cache_dir / "g2_queries.jsonl",
        help="Original G2 query index in JSONL form.",
    )
    parser.add_argument(
        "--g3-query-index",
        type=Path,
        default=cache_dir / "g3_queries.jsonl",
        help="Original G3 query index in JSONL form.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=analysis_dir / "data_samples",
        help="Directory for g1_10.jsonl, g2_10.jsonl, and g3_10.jsonl.",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing default source files from the documented mirrors.",
    )
    parser.add_argument(
        "--give-answer-count",
        type=int,
        default=5,
        help=(
            "Number of give_answer trajectories per group. The remainder are "
            "give_up_and_restart trajectories; use -1 for unstratified sampling."
        ),
    )
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n- " + "\n- ".join(missing))


def download_file(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}\n        -> {destination}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        partial.replace(destination)
    finally:
        if partial.exists() and not destination.exists():
            partial.unlink()


def load_query_index(path: Path) -> set[str]:
    queries: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no non-empty 'text' field")
            queries.add(normalize_query(text))
    return queries


def normalize_query(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s*Begin!\s*$", "", text).strip()
    return text


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, dict) or "from" not in value or "value" not in value:
        raise ValueError("Unexpected conversations value; expected {'from': ..., 'value': ...}")
    roles = as_list(value["from"])
    contents = as_list(value["value"])
    if len(roles) != len(contents):
        raise ValueError("Conversation role/content lengths differ")
    return [
        {"from": str(role), "value": "" if content is None else str(content)}
        for role, content in zip(roles, contents)
    ]


def extract_query(conversations: list[dict[str, str]]) -> str:
    user_messages = [m["value"] for m in conversations if m["from"] == "user"]
    if not user_messages:
        raise ValueError("Conversation has no user message")
    task_messages = [message for message in user_messages if "Begin!" in message]
    return normalize_query(task_messages[0] if task_messages else user_messages[0])


def classify_group(
    query: str,
    original_tools: list[str],
    g2_queries: set[str],
    g3_queries: set[str],
) -> tuple[str, str]:
    in_g2 = query in g2_queries
    in_g3 = query in g3_queries
    if in_g2 and in_g3:
        raise ValueError("A query appears in both G2 and G3 indexes")
    if in_g2:
        return "G2", "exact_query_match:g2_index"
    if in_g3:
        return "G3", "exact_query_match:g3_index"
    if len(original_tools) == 1:
        return "G1", "single_original_tool_and_not_in_g2_g3"
    return "UNKNOWN", "not_in_g2_g3_and_not_single_tool"


def parse_outcome(final_message: str) -> str:
    match = RETURN_TYPE_RE.search(final_message)
    return match.group(1) if match else "unknown"


def collect_candidates(
    parquet_path: Path,
    g2_queries: set[str],
    g3_queries: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    frame = pd.read_parquet(parquet_path)
    required_columns = {"id", "conversations"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Parquet must contain columns: {sorted(required_columns)}")

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics: Counter[str] = Counter()
    seen_completed_queries: set[tuple[str, str]] = set()

    for _, row in frame.iterrows():
        conversations = normalize_conversations(row["conversations"])
        if not conversations:
            diagnostics["empty_conversation"] += 1
            continue

        final = conversations[-1]
        if final["from"] != "assistant" or not FINISH_RE.search(final["value"]):
            diagnostics["intermediate_prefix_skipped"] += 1
            continue

        query = extract_query(conversations)
        system_text = next(
            (message["value"] for message in conversations if message["from"] == "system"),
            "",
        )
        original_tools = ORIGINAL_TOOL_RE.findall(system_text)
        group, classification_basis = classify_group(
            query, original_tools, g2_queries, g3_queries
        )
        if group == "UNKNOWN":
            diagnostics["unknown_group_skipped"] += 1
            continue

        duplicate_key = (group, query)
        if duplicate_key in seen_completed_queries:
            raise ValueError(f"Multiple completed trajectories found for {group}: {query}")
        seen_completed_queries.add(duplicate_key)

        actions = [
            action.strip()
            for message in conversations
            if message["from"] == "assistant"
            for action in ACTION_RE.findall(message["value"])
        ]
        step_match = STEP_RE.match(str(row["id"]))
        pools[group].append(
            {
                "source_record_id": str(row["id"]),
                "query": query,
                "group": group,
                "classification_basis": classification_basis,
                "original_tools": original_tools,
                "tool_count": len(original_tools),
                "action_names": actions,
                "action_count": len(actions),
                "source_step": int(step_match.group(1)) if step_match else None,
                "outcome": parse_outcome(final["value"]),
                "conversations": conversations,
            }
        )
        diagnostics[f"eligible_{group.lower()}"] += 1

    return pools, diagnostics


def choose_samples(
    pool: list[dict[str, Any]],
    count: int,
    give_answer_count: int,
    rng: random.Random,
    group: str,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if len(pool) < count:
        raise ValueError(f"{group} has only {len(pool)} eligible trajectories; need {count}")

    if give_answer_count < 0:
        chosen = rng.sample(pool, count)
    else:
        if give_answer_count > count:
            raise ValueError("--give-answer-count cannot exceed --count")
        give_up_count = count - give_answer_count
        by_outcome: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pool:
            by_outcome[record["outcome"]].append(record)
        if len(by_outcome["give_answer"]) < give_answer_count:
            raise ValueError(f"{group} lacks enough give_answer trajectories")
        if len(by_outcome["give_up_and_restart"]) < give_up_count:
            raise ValueError(f"{group} lacks enough give_up_and_restart trajectories")
        chosen = rng.sample(by_outcome["give_answer"], give_answer_count)
        chosen += rng.sample(by_outcome["give_up_and_restart"], give_up_count)
        rng.shuffle(chosen)

    # Stable presentation order; sampling itself remains controlled by the seed.
    return sorted(chosen, key=lambda item: item["query"])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.download_missing:
        download_file(SOURCE_URLS["trajectory_parquet"], args.trajectory_parquet)
        download_file(SOURCE_URLS["g2_query_index"], args.g2_query_index)
        download_file(SOURCE_URLS["g3_query_index"], args.g3_query_index)
    require_files(
        [args.trajectory_parquet, args.g2_query_index, args.g3_query_index]
    )
    g2_queries = load_query_index(args.g2_query_index)
    g3_queries = load_query_index(args.g3_query_index)
    pools, diagnostics = collect_candidates(
        args.trajectory_parquet, g2_queries, g3_queries
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "count_per_group": args.count,
        "give_answer_count_per_group": args.give_answer_count,
        "source": {
            "trajectory_dataset": "tuandunghcmut/toolbench-v1",
            "trajectory_split": "validation",
            "g2_query_index": "Tool-COLT/ToolBenchG2",
            "g3_query_index": "Tool-COLT/ToolBenchG3",
            "urls": SOURCE_URLS,
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "outputs": {},
    }

    for group in GROUPS:
        rng = random.Random(f"{args.seed}:{group}")
        samples = choose_samples(
            pools[group], args.count, args.give_answer_count, rng, group
        )
        output_records: list[dict[str, Any]] = []
        for number, sample in enumerate(samples, start=1):
            uid = hashlib.sha256(sample["query"].encode("utf-8")).hexdigest()[:12]
            output_records.append(
                {
                    "sample_id": f"{group}_{number:03d}",
                    "sample_uid": uid,
                    **sample,
                    "provenance": manifest["source"],
                }
            )

        output_path = args.output_dir / f"{group.lower()}_{args.count}.jsonl"
        write_jsonl(output_path, output_records)
        outcome_counts = Counter(record["outcome"] for record in output_records)
        manifest["outputs"][group] = {
            "path": str(output_path),
            "records": len(output_records),
            "unique_queries": len({record["query"] for record in output_records}),
            "outcomes": dict(sorted(outcome_counts.items())),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }

    manifest_path = args.output_dir / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
