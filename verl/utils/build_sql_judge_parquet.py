import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import datasets


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"label must be 0/1, got {value}")
    if isinstance(value, str):
        value = value.strip()
        if value in ("0", "1"):
            return int(value)
    raise ValueError(f"invalid label value: {value}")


def build_user_prompt(
    *,
    user_question: str,
    schema: str,
    reference_sql: str,
    predicted_sql: str,
) -> str:
    return f"""You are a strict SQL semantic judge.
Given a user question, a database schema, a reference SQL, and a candidate SQL.
Decide whether the candidate SQL can correctly answer the user question under the schema.
Semantic equivalence is valid even if SQL forms are different.

Output only one final tag:
<answer>YES</answer> or <answer>NO</answer>

User Question:
{user_question}

Reference SQL:
{reference_sql}

Candidate SQL:
{predicted_sql}

Database Schema:
{schema}
"""


def build_one_row(
    *,
    data_source: str,
    pair_index: int,
    row_index: int,
    pair_type: str,
    user_question: str,
    schema: str,
    reference_sql: str,
    predicted_sql: str,
    gold_label: int,
    db_file: str,
) -> dict[str, Any]:
    prompt_text = build_user_prompt(
        user_question=user_question,
        schema=schema,
        reference_sql=reference_sql,
        predicted_sql=predicted_sql,
    )

    extra_info = {
        "index": row_index,
        "pair_index": pair_index,
        "pair_type": pair_type,
        "reference_sql": reference_sql,
        "predicted_sql": predicted_sql,
        "schema": schema,
        "user_question": user_question,
        "gold_label": gold_label,
    }

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "label": gold_label,
        "ability": "sql",
        "db_file": db_file,
        "extra_info": extra_info,
    }


def build_rows_from_pairs(
    pairs: list[dict[str, Any]],
    *,
    data_source: str,
    pos_pos_both: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = 0

    for pair_index, item in enumerate(pairs):
        if not isinstance(item, dict):
            continue

        pair_type = str(item.get("type", "")).strip()
        user_question = str(item.get("question", "")).strip()
        schema = str(item.get("schema", "")).strip()
        sql1 = str(item.get("sql1", "")).strip()
        sql2 = str(item.get("sql2", "")).strip()
        db_file = str(item.get("db_file", "")).strip()

        if not user_question or not schema or not sql1 or not sql2:
            continue

        if pair_type == "pos_neg":
            configs = [
                (sql1, 1),
                (sql2, 0),
            ]
        elif pair_type == "pos_pos":
            configs = [(sql2, 1)]
            if pos_pos_both:
                configs = [
                    (sql1, 1),
                    (sql2, 1),
                ]
        else:
            pair_label = parse_label(item.get("label"))
            configs = [(sql2, pair_label)]

        for predicted_sql, gold_label in configs:
            rows.append(
                build_one_row(
                    data_source=data_source,
                    pair_index=pair_index,
                    row_index=row_index,
                    pair_type=pair_type,
                    user_question=user_question,
                    schema=schema,
                    reference_sql=sql1,
                    predicted_sql=predicted_sql,
                    gold_label=gold_label,
                    db_file=db_file,
                )
            )
            row_index += 1

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
    parser.add_argument("--input", type=str, required=True, help="Input pair JSON path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory containing train/test parquet.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio in [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--data-source", type=str, default="sql_judge_pairs", help="data_source field value.")
    parser.add_argument(
        "--pos-pos-both",
        action="store_true",
        help="If set, pos_pos pair emits two positive rows (sql1 and sql2).",
    )
    parser.add_argument("--indent", type=int, default=2, help="Indent for example json outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig") as f:
        pairs = json.load(f)
    if not isinstance(pairs, list):
        raise ValueError(f"input must be a JSON list, got {type(pairs).__name__}")

    rows = build_rows_from_pairs(
        pairs,
        data_source=args.data_source,
        pos_pos_both=args.pos_pos_both,
    )
    if not rows:
        raise ValueError("no valid rows generated from input pairs")

    train_rows, test_rows = split_rows(rows, test_ratio=args.test_ratio, seed=args.seed)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    template_row = rows[0]

    save_parquet(train_rows, train_path, template_row=template_row)
    save_parquet(test_rows, test_path, template_row=template_row)

    train_pos, train_neg = label_stats(train_rows)
    test_pos, test_neg = label_stats(test_rows)
    print(f"input_pairs={len(pairs)} expanded_rows={len(rows)}")
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
