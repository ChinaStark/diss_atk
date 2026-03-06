import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

# =========================
# Patterns
# =========================
# Strict: must be exactly <think>...</think><answer>YES|NO</answer> (allow whitespace between blocks)
_STRICT_PATTERN = re.compile(
    r"^\s*<think>(?P<think>[\s\S]*?)</think>\s*<answer>\s*(?P<ans>YES|NO)\s*</answer>\s*$",
    flags=re.IGNORECASE,
)

# Loose: just find tags anywhere
_ANSWER_TAG = re.compile(r"<answer>\s*(?P<ans>[\s\S]*?)\s*</answer>", flags=re.IGNORECASE)
_THINK_TAG = re.compile(r"<think>\s*(?P<think>[\s\S]*?)\s*</think>", flags=re.IGNORECASE)


def _split_assistant(solution_str: str) -> Tuple[Optional[str], str]:
    """Strictly follow the given split logic to extract assistant content."""
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1], ""
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant", 1)[1], ""
    return solution_str, ""


def _normalize_answer(raw: str) -> Optional[str]:
    """Normalize <answer> content to YES/NO."""
    if raw is None:
        return None
    s = re.sub(r"\s+", "", raw).upper()
    if not s:
        return None
    m = re.match(r"^(YES|NO)", s)
    if not m:
        return None
    return m.group(1)


def _extract_loose(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (think_text, pred) loosely from tags if present."""
    m_think = _THINK_TAG.search(text)
    think_text = m_think.group("think") if m_think else None

    m_ans = _ANSWER_TAG.search(text)
    pred = _normalize_answer(m_ans.group("ans")) if m_ans else None

    return think_text, pred


@dataclass
class ScoreWeights:
    # 1) format rewards (keep original)
    strict_format: float = 0.5

    # 2) label rewards
    # Softmax-margin reward magnitude. Actual range: [-label_softmax_mag, label_softmax_mag]
    label_softmax_mag: float = 2.0

    # 3) LLM reward
    # LLM score is normalized to [0,1], then mapped to [-1,1] before multiplying this magnitude.
    llm_reward_mag: float = 1.0

    # 4) length reward (difficulty-aware)
    # Ideal len = len_easy + difficulty * (len_hard - len_easy)
    len_easy: int = 192
    len_hard: int = 1024
    len_tolerance_ratio: float = 0.35
    len_mag: float = 0.5
    default_difficulty: float = 0.5


@dataclass
class VLLMOpenAIConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    api_key: Optional[str] = os.getenv("VLLM_API_KEY", None)
    model: str = os.getenv("VLLM_MODEL", "")
    timeout_s: int = 60
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _softmax2(a: float, b: float) -> Tuple[float, float]:
    """Numerically stable 2-class softmax."""
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def _extract_yes_no_pair(
    logits: Optional[Mapping[str, Any] | Tuple[Any, Any]],
    probs: Optional[Mapping[str, Any] | Tuple[Any, Any]],
) -> Tuple[Optional[float], Optional[float], str]:
    """Extract YES/NO evidence in one of two forms:
    1) logits (preferred)
    2) probabilities
    """
    if logits is not None:
        yes, no = None, None
        try:
            yes = _to_float(logits.get("YES", logits.get("yes")))
            no = _to_float(logits.get("NO", logits.get("no")))
        except Exception:
            pass
        if yes is None or no is None:
            try:
                yes = _to_float(logits[0])
                no = _to_float(logits[1])
            except Exception:
                pass
        if yes is not None and no is not None:
            return yes, no, "logits"

    if probs is not None:
        yes, no = None, None
        try:
            yes = _to_float(probs.get("YES", probs.get("yes")))
            no = _to_float(probs.get("NO", probs.get("no")))
        except Exception:
            pass
        if yes is None or no is None:
            try:
                yes = _to_float(probs[0])
                no = _to_float(probs[1])
            except Exception:
                pass
        if yes is not None and no is not None:
            return yes, no, "probs"

    return None, None, "missing"


def _call_llm(
    llm_judge_fn: Optional[Callable[[dict[str, Any]], Any]],
    *,
    payload: dict[str, Any],
) -> Any:
    if llm_judge_fn is not None:
        return llm_judge_fn(payload)

    # Default path: call vLLM OpenAI-compatible API.
    extra_context = payload.get("extra_context") or {}
    vllm_cfg_raw = extra_context.get("vllm_openai") or {}
    cfg = VLLMOpenAIConfig(
        base_url=vllm_cfg_raw.get("base_url", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")),
        api_key=vllm_cfg_raw.get("api_key", os.getenv("VLLM_API_KEY")),
        model=vllm_cfg_raw.get("model", os.getenv("VLLM_MODEL", "")),
        timeout_s=int(vllm_cfg_raw.get("timeout_s", 60)),
        max_tokens=int(vllm_cfg_raw.get("max_tokens", 1024)),
        temperature=float(vllm_cfg_raw.get("temperature", 0.0)),
        top_p=float(vllm_cfg_raw.get("top_p", 0.95)),
    )
    if not cfg.model:
        raise ValueError("VLLM model is empty. Set VLLM_MODEL or llm_extra_context['vllm_openai']['model'].")
    if "messages" not in payload:
        raise ValueError("payload['messages'] is required for vLLM OpenAI chat call")

    request_payload = {
        "model": cfg.model,
        "messages": payload["messages"],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    import requests

    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    response = requests.post(
        url,
        headers=headers,
        json=request_payload,
        timeout=cfg.timeout_s,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()

    return content


def _build_sql_rubric(
    llm_judge_fn: Optional[Callable[[dict[str, Any]], Any]],
    *,
    model_output: str,
    gold_label: int,
    reference_sql: str,
    predicted_sql: str,
    schema: str,
    extra_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = (
        "Build rubric and key_points for SQL evaluation.\n"
        "Return JSON only with keys: rubric, key_points.\n"
        f"gold_label: {gold_label}\n"
        f"model_output: {model_output}\n"
        f"reference_sql: {reference_sql}\n"
        f"predicted_sql: {predicted_sql}\n"
        f"schema: {schema}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You are a strict SQL evaluator. Return valid JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        "extra_context": extra_context or {},
    }
    return _call_llm(llm_judge_fn, payload=payload)


def _judge_sql_by_rubric(
    llm_judge_fn: Optional[Callable[[dict[str, Any]], Any]],
    *,
    model_output: str,
    gold_label: int,
    reference_sql: str,
    predicted_sql: str,
    schema: str,
    rubric: Any,
    key_points: Any,
    extra_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = (
        "Judge predicted_sql with rubric and key_points.\n"
        "Return JSON only with keys: score, difficulty, reason.\n"
        f"gold_label: {gold_label}\n"
        f"model_output: {model_output}\n"
        f"reference_sql: {reference_sql}\n"
        f"predicted_sql: {predicted_sql}\n"
        f"schema: {schema}\n"
        f"rubric: {rubric}\n"
        f"key_points: {key_points}"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You are a strict SQL evaluator. Return valid JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        "extra_context": extra_context or {},
    }
    return _call_llm(llm_judge_fn, payload=payload)


def _call_llm_judge(
    llm_judge_fn: Optional[Callable[[dict[str, Any]], Any]],
    *,
    model_output: str,
    gold_label: int,
    reference_sql: Optional[str],
    predicted_sql: Optional[str],
    schema: Optional[str],
    extra_context: Optional[dict[str, Any]],
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Two-step LLM pipeline:
    1) build rubric/key_points
    2) judge by rubric and return score/difficulty/reason
    """
    meta: dict[str, Any] = {"enabled": False, "todo": []}
    empty_result = {
        "rubric": None,
        "key_points": None,
        "score": None,
        "difficulty": None,
        "reason": None,
    }

    missing_fields = []
    if reference_sql is None:
        missing_fields.append("reference_sql")
    if predicted_sql is None:
        missing_fields.append("predicted_sql")
    if schema is None:
        missing_fields.append("schema")
    if missing_fields:
        meta["todo"].append(f"TODO: provide {', '.join(missing_fields)} for LLM judging")
        return empty_result, meta

    try:
        rubric_result = _build_sql_rubric(
            llm_judge_fn,
            model_output=model_output,
            gold_label=gold_label,
            reference_sql=reference_sql,
            predicted_sql=predicted_sql,
            schema=schema,
            extra_context=extra_context,
        )
        rubric = rubric_result.get("rubric")
        key_points = rubric_result.get("key_points")

        judge_result = _judge_sql_by_rubric(
            llm_judge_fn,
            model_output=model_output,
            gold_label=gold_label,
            reference_sql=reference_sql,
            predicted_sql=predicted_sql,
            schema=schema,
            rubric=rubric,
            key_points=key_points,
            extra_context=extra_context,
        )

        result = {
            "rubric": rubric,
            "key_points": key_points,
            "score": judge_result.get("score"),
            "difficulty": judge_result.get("difficulty"),
            "reason": judge_result.get("reason"),
        }
        meta.update({"enabled": True, "rubric_raw": rubric_result, "judge_raw": judge_result})
        return result, meta
    except Exception as exc:  # noqa: BLE001
        meta["error"] = str(exc)
        meta["todo"].append(
            "TODO: ensure vLLM(OpenAI format) or llm_judge_fn returns dict with rubric/key_points/score/difficulty/reason"
        )
        return empty_result, meta


def compute_score(
    model_output: str,
    gold_label: int,
    resp_len: int,
    # Backward-compat only. Deprecated in difficulty-aware length mode.
    L: Optional[int] = None,
    # TODO(user): pass YES/NO logits/probs from the policy model forward pass.
    yes_no_logits: Optional[Mapping[str, Any] | Tuple[Any, Any]] = None,
    yes_no_probs: Optional[Mapping[str, Any] | Tuple[Any, Any]] = None,
    # TODO(user): pass SQL task fields. For LLM call, either pass llm_judge_fn or configure llm_extra_context['vllm_openai'].
    reference_sql: Optional[str] = None,
    predicted_sql: Optional[str] = None,
    schema: Optional[str] = None,
    llm_judge_fn: Optional[Callable[[dict[str, Any]], Any]] = None,
    llm_extra_context: Optional[dict[str, Any]] = None,
    # Optional external difficulty override in [0,1].
    difficulty_score: Optional[float] = None,
    weights: Optional[ScoreWeights] = None,
    return_breakdown: bool = False,
) -> Any:
    """Compute reward = format + label(softmax) + llm + length(difficulty-aware).

    Priority:
    1) Format reward: original strict format check.
    2) Label reward: softmax margin over YES/NO probabilities.
    3) LLM reward: use llm_judge_fn if provided, else call vLLM OpenAI-compatible API.
    4) Length reward: shaped by difficulty (from arg, else from LLM, else default).

    If YES/NO logits/probs are absent, label reward is 0 (no fallback) and TODO hints are returned in meta.
    """
    if weights is None:
        weights = ScoreWeights()

    gold = "YES" if gold_label == 1 else "NO"

    processed, err = _split_assistant(model_output)
    text = processed if processed is not None else model_output

    # Strict match
    m_strict = _STRICT_PATTERN.match(text)
    strict_ok = m_strict is not None

    # Loose extraction (independent of strict)
    _, pred = _extract_loose(text)

    # -------------------------
    # 1) Format score
    # -------------------------
    r_format = weights.strict_format if strict_ok else 0.0

    # -------------------------
    # 2) Label score
    # -------------------------
    yes_raw, no_raw, yn_source = _extract_yes_no_pair(yes_no_logits, yes_no_probs)
    label_meta: dict[str, Any] = {"source": yn_source, "todo": []}

    if yes_raw is not None and no_raw is not None:
        if yn_source == "logits":
            p_yes, p_no = _softmax2(yes_raw, no_raw)
        else:
            s = max(1e-12, yes_raw + no_raw)
            p_yes, p_no = yes_raw / s, no_raw / s

        margin = (p_yes - p_no) if gold_label == 1 else (p_no - p_yes)
        r_label = weights.label_softmax_mag * margin
        label_meta.update(
            {
                "p_yes": p_yes,
                "p_no": p_no,
                "margin": margin,
            }
        )
    else:
        # No fallback: label reward only comes from YES/NO logits(or probs) softmax margin.
        r_label = 0.0
        label_meta["parsed_answer"] = pred
        label_meta["todo"].append("TODO: pass yes_no_logits or yes_no_probs for softmax label reward")

    # -------------------------
    # 3) LLM score
    # -------------------------
    llm_result, llm_meta = _call_llm_judge(
        llm_judge_fn,
        model_output=text,
        gold_label=gold_label,
        reference_sql=reference_sql,
        predicted_sql=predicted_sql,
        schema=schema,
        extra_context=llm_extra_context,
    )
    llm_meta["result"] = llm_result

    llm_score_raw = _to_float(llm_result.get("score"))
    if llm_score_raw is None:
        r_llm = 0.0
        llm_score_01 = None
    else:
        # Keep returned score raw, only convert here for reward mapping.
        if 0.0 <= llm_score_raw <= 1.0:
            llm_score_01 = llm_score_raw
        elif 0.0 <= llm_score_raw <= 10.0:
            llm_score_01 = llm_score_raw / 10.0
        elif 0.0 <= llm_score_raw <= 100.0:
            llm_score_01 = llm_score_raw / 100.0
        else:
            llm_score_01 = llm_score_raw
        llm_score_01 = _clip(llm_score_01, 0.0, 1.0)
        # [0,1] -> [-1,1]
        r_llm = weights.llm_reward_mag * (2.0 * llm_score_01 - 1.0)

    # -------------------------
    # 4) Length score (difficulty-aware)
    # -------------------------
    arg_difficulty_raw = _to_float(difficulty_score)
    llm_difficulty_raw = _to_float(llm_result.get("difficulty"))
    if arg_difficulty_raw is not None:
        difficulty_raw = arg_difficulty_raw
        difficulty_source = "arg"
    elif llm_difficulty_raw is not None:
        difficulty_raw = llm_difficulty_raw
        difficulty_source = "llm"
    elif L is not None:
        # Compatibility fallback: map old threshold L to a midpoint-style difficulty estimate.
        difficulty_01 = _clip((float(L) - float(weights.len_easy)) / max(1.0, float(weights.len_hard - weights.len_easy)), 0.0, 1.0)
        difficulty_raw = None
        difficulty_source = "legacy_L"
    else:
        difficulty_01 = _clip(weights.default_difficulty, 0.0, 1.0)
        difficulty_raw = None
        difficulty_source = "default"

    if difficulty_source in ("arg", "llm"):
        # Keep returned difficulty raw, only convert here for length shaping.
        if 0.0 <= difficulty_raw <= 1.0:
            difficulty_01 = difficulty_raw
        elif 1.0 <= difficulty_raw <= 5.0:
            difficulty_01 = (difficulty_raw - 1.0) / 4.0
        elif 0.0 <= difficulty_raw <= 10.0:
            difficulty_01 = difficulty_raw / 10.0
        elif 0.0 <= difficulty_raw <= 100.0:
            difficulty_01 = difficulty_raw / 100.0
        else:
            difficulty_01 = difficulty_raw
        difficulty_01 = _clip(difficulty_01, 0.0, 1.0)

    ideal_len = int(round(weights.len_easy + difficulty_01 * (weights.len_hard - weights.len_easy)))
    ideal_len = max(1, ideal_len)
    tol = max(16, int(round(ideal_len * max(1e-6, weights.len_tolerance_ratio))))
    deviation = abs(resp_len - ideal_len)
    closeness = 1.0 - min(1.0, deviation / tol)
    # closeness in [0,1] -> length reward in [-len_mag, +len_mag]
    r_len = weights.len_mag * (2.0 * closeness - 1.0)

    total = r_format + r_label + r_llm + r_len

    summary = {
        "total_score": total,
        "detail_score": {
            "r_format": r_format,
            "r_label": r_label,
            "r_llm": r_llm,
            "r_len": r_len,
        },
        "difficulty": {
            "score_01": difficulty_01,
            "source": difficulty_source,
            "ideal_len": ideal_len,
        },
        "llm_resp": processed,
        "ll_resp_len": resp_len,
        "label_meta": label_meta,
        "llm_meta": llm_meta,
    }

    logger_score = logging.getLogger("score_logger")
    logger_score.info(json.dumps(summary, ensure_ascii=False))

    if not return_breakdown:
        return float(total)

    return {
        "total": float(total),
        "format": float(r_format),
        "label": float(r_label),
        "llm": float(r_llm),
        "length": float(r_len),
        "meta": {
            "strict_ok": bool(strict_ok),
            "pred": pred,
            "gold": gold,
            "token_len": int(resp_len),
            "header_error": err,
            "label_meta": label_meta,
            "llm_meta": llm_meta,
            "difficulty": {
                "score_01": float(difficulty_01),
                "source": difficulty_source,
                "ideal_len": int(ideal_len),
                "tolerance": int(tol),
            },
        },
    }


# -------------------------
# Quick self-checks (optional)
# -------------------------
if __name__ == "__main__":
    out = "Assistant:<think>reasoning</think><answer>YES</answer>"

    # 1) no logits mode (label reward = 0, no fallback)
    print(compute_score(out, gold_label=1, resp_len=120, return_breakdown=True))

    # 2) softmax-label mode
    print(
        compute_score(
            out,
            gold_label=1,
            resp_len=120,
            yes_no_logits={"YES": 3.2, "NO": 0.3},
            difficulty_score=0.2,
            return_breakdown=True,
        )
    )
