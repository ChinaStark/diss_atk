import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import datasets


SQL_CHECK_PROMPT = """Task Overview:
You are a data science expert. Below, you are provided with a database schema, natural language question and a SQL query written by the developer. Your task is to understand the schema and determine whether the SQL query written by the developer can correctly answer the natural language question.

Database Engine:
SQLite

Database Schema:
{{SCHEMA}}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

User Question:
{{QUESTION}}

Developer SQL:
{{FILTER_SQL}}

Instructions:
- First, carefully understand the question and identify exactly what information the question expects (columns, filters, aggregations, ordering, limit, and any other constraints).
- Then, analyze the provided SQL and determine whether it returns exactly what the question asks for — no more, no less.
- Even small mismatches should be considered incorrect (e.g., extra columns, missing filters, wrong table joins, wrong ordering, wrong limit, wrong logic, not null, etc.).
- Note that while the reasoning process and you decision need to be enclosed within <think> </think> and <answer> </answer> tags respectively, this should not affect the quality of you decision.

Output format Example:
<think>
(Your step-by-step reasoning in natural language here)
</think>
<answer>
YES or NO
</answer>

Take a deep breath and think step by step to make important decision.
"""


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"label must be 0/1, got {value}")
    if isinstance(value, str):
        text = value.strip()
        if text in ("0", "1"):
            return int(text)
    raise ValueError(f"invalid label value: {value}")


def render_sql_check_prompt(schema: str, question: str, predicted_sql: str) -> str:
    return (
        SQL_CHECK_PROMPT.replace("{{SCHEMA}}", schema)
        .replace("{{QUESTION}}", question)
        .replace("{{FILTER_SQL}}", predicted_sql)
    )


def build_one_row(*, item: dict[str, Any], row_index: int, data_source: str) -> dict[str, Any]:
    predicted_sql = str(item["sql"]).strip()
    schema = str(item["schema"]).strip()
    user_question = str(item["question"]).strip()
    gold_label = parse_label(item["label"])
    db_file = str(item["db_file"]).strip()

    prompt_text = render_sql_check_prompt(
        schema=schema,
        question=user_question,
        predicted_sql=predicted_sql,
    )

    extra_info = {
        "gold_label": gold_label,
        "predicted_sql": predicted_sql,
        "schema": schema,
        "user_question": user_question,
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "label": gold_label,
        "ability": "sql",
        "db_file": db_file,
        "exec_idx": item["exec_idx"],
        "extra_info": extra_info,
        "ground_truth": item["ground_truth"],
        "sample_rule": item["sample_rule"],
        "embed_similarity": item["embed_similarity"],
        "type": item["type"],
    }


def build_rows_from_items(items: list[dict[str, Any]], *, data_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        rows.append(build_one_row(item=item, row_index=row_index, data_source=data_source))
    return rows


def split_rows(rows: list[dict[str, Any]], *, test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("--test-ratio must be in [0, 1).")
    if not rows:
        return [], []

    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    test_size = int(round(len(shuffled) * test_ratio))
    if test_ratio > 0.0 and test_size == 0:
        test_size = 1
    if test_size >= len(shuffled):
        test_size = len(shuffled) - 1

    test_rows = shuffled[:test_size]
    train_rows = shuffled[test_size:]
    return train_rows, test_rows


def save_parquet(rows: list[dict[str, Any]], output_path: Path, *, template_row: dict[str, Any] | None) -> None:
    if rows:
        ds = datasets.Dataset.from_list(rows)
    else:
        if template_row is None:
            raise ValueError(f"cannot save empty parquet without template: {output_path}")
        ds = datasets.Dataset.from_list([template_row]).select([])
    ds.to_parquet(str(output_path))


def label_stats(rows: list[dict[str, Any]]) -> tuple[int, int]:
    pos = 0
    neg = 0
    for row in rows:
        if int(row["label"]) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/test parquet for SQL semantic judge.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON path (single-item SQL records).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory containing train/test parquet.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio in [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--data-source", type=str, default="sql_judge_single_items", help="data_source field value.")
    parser.add_argument("--indent", type=int, default=2, help="Indent for example json outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"input must be a JSON list, got {type(items).__name__}")

    rows = build_rows_from_items(items, data_source=args.data_source)
    if not rows:
        raise ValueError("no valid rows generated from input")

    train_rows, test_rows = split_rows(rows, test_ratio=args.test_ratio, seed=args.seed)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    template_row = rows[0]

    save_parquet(train_rows, train_path, template_row=template_row)
    save_parquet(test_rows, test_path, template_row=template_row)

    train_pos, train_neg = label_stats(train_rows)
    test_pos, test_neg = label_stats(test_rows)
    print(f"input_items={len(items)} rows={len(rows)}")
    print(f"train={len(train_rows)} (pos={train_pos}, neg={train_neg}) -> {train_path}")
    print(f"test={len(test_rows)} (pos={test_pos}, neg={test_neg}) -> {test_path}")

    train_example_path = output_dir / "train_example.json"
    test_example_path = output_dir / "test_example.json"
    with train_example_path.open("w", encoding="utf-8") as f:
        json.dump(train_rows[0], f, ensure_ascii=False, indent=args.indent)
    if test_rows:
        with test_example_path.open("w", encoding="utf-8") as f:
            json.dump(test_rows[0], f, ensure_ascii=False, indent=args.indent)
    else:
        with test_example_path.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=args.indent)

    print(f"saved examples: {train_example_path}, {test_example_path}")


if __name__ == "__main__":
    main()
