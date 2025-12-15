import re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

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

# Remove any XML-ish tags for length fallback
_TAGS = re.compile(r"</?[^>]+>")


def _split_assistant(solution_str: str) -> Tuple[Optional[str], str]:
    """Strictly follow the given split logic to extract assistant content."""
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1], ""
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant", 1)[1], ""
    return solution_str, ""


def _normalize_answer(raw: str) -> Optional[str]:
    """Normalize <answer> content to YES/NO.

    Requirement: delete extra newlines/spaces, etc.
    We accept variants like: " YES\n" or "NO  ", and also tolerate punctuation like "YES.".
    """
    if raw is None:
        return None
    # remove all whitespace (space/tab/newline) per requirement
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
    # format rewards
    strict_format: float = 0.5
    has_answer_tag_only: float = 0.1

    # label rewards
    label_correct_strict: float = 2.0
    label_correct_loose: float = 0.4  # <=  when format is wrong but answer is correct

    # length reward magnitude (max absolute)
    len_mag: float = 0.5

    # If neither strict nor <answer> exists, apply a mild length penalty to discourage very long junk.
    no_pred_len_penalty_mag: float = 0.25

    # Within this many tokens, do NOT apply length penalty.
    # Penalties (and the decay of bonus) start only AFTER free_len.
    free_len: int = 512

    # Penalty = -over_len_penalty_mag * min((length - L) / L, 1.0)
    over_len_penalty_mag: float = 0.5
import os
import json
import logging
from logging.handlers import RotatingFileHandler

def get_score_logger(log_file: str = "score.log", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or reuse) a file logger.
    - No console output
    - Rotating file to avoid huge logs
    """
    logger = logging.getLogger("score_logger")
    logger.setLevel(level)
    logger.propagate = False  # IMPORTANT: prevents printing to root logger/console

    abs_path = os.path.abspath(log_file)
    # Avoid adding duplicate handlers (common when this code is imported multiple times)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path:
            return logger

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    fh = RotatingFileHandler(
        abs_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s"
    ))
    logger.addHandler(fh)
    return logger


# 建议：模块级只初始化一次
_SCORE_LOGGER = get_score_logger("logs/rollouts.log")

def compute_score(
    model_output: str,
    gold_label: int,
    resp_len: int,
    L: int = 512,
    weights: ScoreWeights = ScoreWeights(),
    return_breakdown: bool = False,
) -> Any:
    """Compute reward with separated components.

    Changes vs original:
      1) Score separation: returns {format, label, length, total, meta} by default.
      2) If strict format fails but <answer> exists and is correct: give a small label reward.
      3) Matching uses ONLY <answer> tag content (whitespace/newlines removed).
      4) Length reward computed robustly even when tags/format are missing.

    Args:
      model_output: full model output (may include header etc.)
      gold_label: 1 -> YES, 0 -> NO
      L: length normalization threshold
      weights: tweakable reward weights
      return_breakdown: if False, return total float only
    """

    gold = "YES" if gold_label == 1 else "NO"

    processed, err = _split_assistant(model_output)
    text = processed if processed is not None else model_output

    # Strict match
    m_strict = _STRICT_PATTERN.match(text)
    strict_ok = m_strict is not None

    # Loose extraction (independent of strict)
    think_text, pred = _extract_loose(text)

    # -------------------------
    # 1) Format score
    # -------------------------
    if strict_ok:
        r_format = weights.strict_format
    else:
        r_format = 0.0

    # -------------------------
    # 2) Label score
    # -------------------------
    if pred is None:
        r_label = 0.0
        correct = False
    else:
        correct = (pred == gold)
        if correct and strict_ok:
            r_label = weights.label_correct_strict
        elif correct:
            r_label = weights.label_correct_loose
        else:
            r_label = 0.0

    # -------------------------
    # 3) Length score
    # -------------------------

    if r_label > 0.0:  
        r_len = 0.0
    else:
        if resp_len > L:
            # penalize proportional to how much it exceeds L
            over = resp_len - L
            r_len = -weights.len_mag * min(1.0, over / max(1, L))
        else:
            # length <= L → no penalty
            r_len = 0.0

    total = r_format + r_label + r_len

    summary = {
        "total_score": r_format + r_label + r_len,
        "detail_score": {
            "r_format": r_format,
            "r_label": r_label,
            "r_len": r_len,
        },
        "llm_resp": processed,   # 或 processed["pred"] 根据你结构改
        "ll_resp_len": resp_len
    }
    
    _SCORE_LOGGER.info(json.dumps(summary, ensure_ascii=False))
    if not return_breakdown:
        return float(total)

    return {
        "total": float(total),
        "format": float(r_format),
        "label": float(r_label),
        "length": float(r_len),
        "meta": {
            "strict_ok": bool(strict_ok),
            "pred": pred,
            "gold": gold,
            "token_len": int(resp_len),
            "header_error": err,
        },
    }


# -------------------------
# Quick self-checks (optional)
# -------------------------
if __name__ == "__main__":
    out1 = "Assistant:<think>hi"+("x" * 1200) + "</think><answer>YES</answer>"
    print(out1)
    print(compute_score(out1, 1))

    out2 = "Assistant: blah blah <answer>  YES  \n</answer>"  # format wrong but correct answer
    print(compute_score(out2, 1))

    out3 = "Assistant: <think>lots of text  <answer>NO</answer>"  # loose ok
    print(compute_score(out3, 1))

    out4 = "Assistant: no tags but very long " + ("x " * 3000)
    print(compute_score(out4, 1))
