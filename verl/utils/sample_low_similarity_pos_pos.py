import argparse
import json
import math
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample lowest-similarity N pos_pos pairs using local Qwen embedding model."
    )
    parser.add_argument("--input", type=str, required=True, help="Input pair JSON path.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--n", type=int, required=True, help="Number of lowest-similarity pos_pos pairs to keep.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        help="Local model path or HuggingFace model id.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Local embedding batch size.")
    parser.add_argument("--max-length", type=int, default=2048, help="Tokenizer max length.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cuda/cpu.")
    parser.add_argument("--indent", type=int, default=2, help="Output JSON indent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input must be a JSON list, got {type(data).__name__}")

    pos_pos_items: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(data):
        if isinstance(item, dict) and item.get("type") == "pos_pos":
            pos_pos_items.append((idx, item))

    if not pos_pos_items:
        raise ValueError("No pos_pos pairs found in input.")

    unique_sqls: list[str] = []
    seen_sqls: set[str] = set()
    for _, item in pos_pos_items:
        sql1 = str(item.get("sql1", ""))
        sql2 = str(item.get("sql2", ""))
        if sql1 not in seen_sqls:
            seen_sqls.add(sql1)
            unique_sqls.append(sql1)
        if sql2 not in seen_sqls:
            seen_sqls.add(sql2)
            unique_sqls.append(sql2)

    print(f"loaded items={len(data)} pos_pos={len(pos_pos_items)} unique_sql={len(unique_sqls)}")
    print(f"embedding model={args.model} device={args.device} max_length={args.max_length}")

    text_to_embedding = encode_texts_local(
        unique_sqls,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    scored_pos_pos: list[tuple[float, int, dict[str, Any]]] = []
    for original_idx, item in pos_pos_items:
        sql1 = str(item.get("sql1", ""))
        sql2 = str(item.get("sql2", ""))
        sim = cosine_similarity(text_to_embedding[sql1], text_to_embedding[sql2])
        scored = dict(item)
        scored["embed_similarity"] = sim
        scored_pos_pos.append((sim, original_idx, scored))

    scored_pos_pos.sort(key=lambda x: x[0])  # lowest similarity first
    keep_n = min(args.n, len(scored_pos_pos))
    selected = scored_pos_pos[:keep_n]
    selected_idx = {idx for _, idx, _ in selected}
    idx_to_scored = {idx: item for _, idx, item in selected}

    output_data: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "pos_pos":
            output_data.append(item)
            continue
        if idx in selected_idx:
            output_data.append(idx_to_scored[idx])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=args.indent)

    selected_sims = [sim for sim, _, _ in selected]
    print(
        f"selected_pos_pos={keep_n}/{len(pos_pos_items)} "
        f"min_sim={min(selected_sims):.6f} max_sim={max(selected_sims):.6f}"
    )
    print(f"output_items={len(output_data)} saved_to={output_path}")


if __name__ == "__main__":
    main()
