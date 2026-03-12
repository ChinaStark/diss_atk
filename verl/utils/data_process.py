import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


SCHEMA_PATTERN = re.compile(
    r"Database Schema:\s*(.*?)\s*This schema describes the database's structure",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_schema(prompt: Any) -> str:
    """Extract schema text between 'Database Schema:' and fixed tail sentence."""
    if not isinstance(prompt, str):
        return ""
    match = SCHEMA_PATTERN.search(prompt)
    if match is None:
        return ""
    return match.group(1).strip()


def sql_sfx(sql: Any) -> str:
    """Normalize SQL for intra-group dedup: strip + lowercase."""
    if sql is None:
        return ""
    return str(sql).strip().lower()


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def group_key(item: dict[str, Any]) -> int:
    """idx // 8 is one question group."""
    return to_int(item.get("idx"), 0) // 8


def dedup_group_items(group_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by (pred_sql_sfx, ground_truth_sql_sfx) within one group."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in group_items:
        key = (item["pred_sql_sfx"], item["ground_truth_sql_sfx"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_pair(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: int,
    pair_type: str,
) -> dict[str, Any]:
    return {
        "sql1": left.get("pred_sql", ""),
        "sql2": right.get("pred_sql", ""),
        "label": int(label),
        "type": pair_type,
        "db_file": left.get("db_file") or right.get("db_file", ""),
        "question": left.get("question") or right.get("question", ""),
        "schema": left.get("schema") or right.get("schema", ""),
    }


def process_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Processing pipeline:
    1) extract schema from prompt
    2) group by idx // 8
    3) drop correctness in {2, 3}
    4) create pred_sql_sfx/ground_truth_sql_sfx and deduplicate per group
    5) build pos_neg and pos_pos pairs
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for raw in items:
        correctness = to_int(raw.get("correctness"))
        if correctness in (2, 3):
            continue

        item = dict(raw)
        item["correctness"] = correctness
        item["schema"] = extract_schema(item.get("prompt"))
        item["pred_sql_sfx"] = sql_sfx(item.get("pred_sql"))
        item["ground_truth_sql_sfx"] = sql_sfx(item.get("ground_truth"))

        grouped[group_key(item)].append(item)

    output_pairs: list[dict[str, Any]] = []

    for _, group_items in grouped.items():
        deduped = dedup_group_items(group_items)
        pos = [x for x in deduped if x.get("correctness") == 1]
        neg = [x for x in deduped if x.get("correctness") == 0]

        # pos-neg: count = len(pos) * len(neg)
        for p in pos:
            for n in neg:
                output_pairs.append(build_pair(p, n, label=0, pair_type="pos_neg"))

        # pos-pos: choose distinct positive pairs
        for p1, p2 in combinations(pos, 2):
            output_pairs.append(build_pair(p1, p2, label=1, pair_type="pos_pos"))

    return output_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NL2SQL pairwise data from raw list JSON.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path (list of items).")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent for output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Input must be a JSON list, got: {type(data).__name__}")

    pairs = process_items(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=args.indent)

    pos_neg_count = sum(1 for x in pairs if x.get("type") == "pos_neg")
    pos_pos_count = sum(1 for x in pairs if x.get("type") == "pos_pos")
    print(f"done. total_pairs={len(pairs)} pos_neg={pos_neg_count} pos_pos={pos_pos_count}")
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
