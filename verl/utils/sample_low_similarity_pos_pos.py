import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError(f"Embedding dim mismatch: {len(vec1)} vs {len(vec2)}")
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts_local(
    texts: list[str],
    *,
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> dict[str, list[float]]:
    if not texts:
        return {}

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(resolved_device)
    model.eval()

    text_to_embedding: dict[str, list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(resolved_device) for k, v in inputs.items()}
            outputs = model(**inputs)

            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                vectors = outputs.text_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vectors = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                vectors = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                vectors = mean_pooling(outputs[0], inputs["attention_mask"])
            else:
                raise ValueError("Model output does not contain text_embeds/pooler_output/last_hidden_state")

            vectors = F.normalize(vectors, p=2, dim=-1).detach().cpu()
            for i, txt in enumerate(batch):
                text_to_embedding[txt] = vectors[i].tolist()
            print(f"encoded {min(start + len(batch), len(texts))}/{len(texts)}")

    return text_to_embedding


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_sql_text(sql: str) -> str:
    return " ".join(sql.replace("\n", " ").replace("\r", " ").split())


def resolve_group_id(item: dict[str, Any], group_size: int) -> Any:
    if "group_id" in item:
        return item["group_id"]
    idx_val = item.get("exec_idx", item.get("idx"))
    idx = to_int(idx_val, None)
    if idx is not None:
        if group_size > 0:
            return idx // group_size
        return idx
    return f"{item.get('db_file', '')}|{item.get('question', '')}"


def resolve_label(item: dict[str, Any]) -> int | None:
    if "correctness" in item:
        return to_int(item["correctness"], None)
    if "label" in item:
        return to_int(item["label"], None)
    if "type" in item:
        t = str(item["type"]).strip().lower()
        if t in {"neg", "negative", "pos_neg"}:
            return 0
        if t in {"pos", "positive", "pos_pos"}:
            return 1
    return None


def resolve_candidate_sql(item: dict[str, Any]) -> str:
    for key in ("sql", "pred_sql", "sql1", "predicted_sql"):
        value = item.get(key)
        if value is not None:
            text = normalize_sql_text(str(value).strip())
            if text:
                return text
    return ""


def resolve_ground_truth_sql(item: dict[str, Any]) -> str:
    for key in ("ground_truth", "reference_sql", "gold_sql", "ground_truth_sql", "sql1"):
        value = item.get(key)
        if value is not None:
            text = normalize_sql_text(str(value).strip())
            if text:
                return text
    return ""


def build_similarity_prompt(question: str, sql: str) -> str:
    # This prompt is used only for semantic embedding of SQL intent.
    return f"""Task: Encode SQL for semantic similarity.
Focus on SQL intent: selected fields, tables, joins, filters, aggregation, grouping, ordering, limit.
Ignore formatting differences (line breaks, indentation, alias style).
Question Context: {question}
SQL Query: {sql}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="For each group, select one negative with highest similarity and one positive with lowest similarity."
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument(
        "--n",
        type=int,
        default=-1,
        help="Max number of groups to sample; <=0 means all groups.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Group size for deriving group_id by exec_idx/idx // group_size.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"),
        help="Local model path or HuggingFace model id.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Local embedding batch size.")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenizer max length.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu.")
    parser.add_argument("--indent", type=int, default=2, help="Output JSON indent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a JSON list, got {type(data).__name__}")

    records: list[dict[str, Any]] = []
    unique_texts: list[str] = []
    seen_texts: set[str] = set()

    for original_idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        label = resolve_label(item)
        if label not in (0, 1):
            continue

        candidate_sql = resolve_candidate_sql(item)
        ground_truth_sql = resolve_ground_truth_sql(item)
        if not candidate_sql or not ground_truth_sql:
            continue

        question = str(item.get("question", "")).strip()
        group_id = resolve_group_id(item, args.group_size)

        gt_text = build_similarity_prompt(question=question, sql=ground_truth_sql)
        cand_text = build_similarity_prompt(question=question, sql=candidate_sql)

        if gt_text not in seen_texts:
            seen_texts.add(gt_text)
            unique_texts.append(gt_text)
        if cand_text not in seen_texts:
            seen_texts.add(cand_text)
            unique_texts.append(cand_text)

        records.append(
            {
                "original_idx": original_idx,
                "group_id": group_id,
                "label": label,
                "item": item,
                "ground_truth_sql": ground_truth_sql,
                "candidate_sql": candidate_sql,
                "gt_text": gt_text,
                "cand_text": cand_text,
            }
        )

    if not records:
        raise ValueError("No valid items found. Need label/correctness + candidate sql + ground_truth sql.")

    print(
        f"loaded items={len(data)} valid_records={len(records)} "
        f"groups={len({r['group_id'] for r in records})} unique_texts={len(unique_texts)}"
    )
    print(f"embedding model={args.model} device={args.device} max_length={args.max_length}")

    text_to_embedding = encode_texts_local(
        unique_texts,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        sim = cosine_similarity(text_to_embedding[rec["gt_text"]], text_to_embedding[rec["cand_text"]])
        rec["embed_similarity"] = sim
        grouped[rec["group_id"]].append(rec)

    selected_records: list[dict[str, Any]] = []
    group_ids = list(grouped.keys())
    if args.n > 0:
        group_ids = group_ids[: args.n]

    for group_id in group_ids:
        group_records = grouped[group_id]
        neg_records = [r for r in group_records if r["label"] == 0]
        pos_records = [r for r in group_records if r["label"] == 1]

        if neg_records:
            # For negatives: choose the one with highest similarity to ground-truth SQL.
            best_neg = max(neg_records, key=lambda r: (r["embed_similarity"], -r["original_idx"]))
            best_neg["sample_rule"] = "neg_max_similarity"
            selected_records.append(best_neg)

        if pos_records:
            # For positives: choose the one with lowest similarity to ground-truth SQL.
            best_pos = min(pos_records, key=lambda r: (r["embed_similarity"], r["original_idx"]))
            best_pos["sample_rule"] = "pos_min_similarity"
            selected_records.append(best_pos)

    selected_records.sort(key=lambda r: r["original_idx"])

    output_data: list[dict[str, Any]] = []
    for rec in selected_records:
        out_item = dict(rec["item"])
        out_item["embed_similarity"] = float(rec["embed_similarity"])
        out_item["sample_rule"] = rec["sample_rule"]
        if "ground_truth" not in out_item:
            out_item["ground_truth"] = rec["ground_truth_sql"]
        output_data.append(out_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=args.indent)

    selected_pos = sum(1 for x in output_data if resolve_label(x) == 1)
    selected_neg = sum(1 for x in output_data if resolve_label(x) == 0)
    sims = [x["embed_similarity"] for x in output_data]
    print(
        f"selected_groups={len(group_ids)} selected_items={len(output_data)} "
        f"pos={selected_pos} neg={selected_neg}"
    )
    if sims:
        print(f"similarity_range=[{min(sims):.6f}, {max(sims):.6f}]")
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
