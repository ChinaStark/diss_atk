import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple
from pydantic import BaseModel

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

# LLM prompts are intentionally fixed in this file.
_RUBRIC_SYSTEM_PROMPT = f"""You are a strict SQL evaluator. Return valid JSON only."""
_JUDGE_SYSTEM_PROMPT = f"""You are a strict SQL evaluator. Return valid JSON only."""
_DIFFICULTY_LEVELS = {"easy", "medium", "hard"}


class _RubricResponse(BaseModel):
    rubric: Any
    key_points: Any


class _JudgeResponse(BaseModel):
    score: float
    difficulty: Literal["easy", "medium", "hard"]
    reason: str


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
    strict_format: float = 2.0

    # 2) label rewards
    # Softmax-margin reward magnitude. Actual range: [0, label_softmax_mag]
    label_softmax_mag: float = 2.0

    # 3) LLM reward
    # LLM score is normalized to [0,1], then mapped to [0, llm_reward_mag].
    llm_reward_mag: float = 2.0

    # 4) length reward in token-length space (difficulty level: easy/medium/hard)
    medium_len: int = 2048  # base length t used by gaussian length rewards
    len_tolerance_ratio: float = 0.5
    len_mag: float = 2.0


@dataclass
class VLLMOpenAIConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode"))
    api_key: Optional[str] = os.getenv("VLLM_API_KEY", os.getenv("DASHSCOPE_API_KEY"))
    model: str = os.getenv("VLLM_MODEL", os.getenv("QWEN_MODEL", "qwen3-coder-plus"))
    timeout_s: int = 60
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _softmax2(a: float, b: float) -> Tuple[float, float]:
    """Numerically stable 2-class softmax."""
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def _compute_length_reward(resp_token_len: int, difficulty_level: str, weights: ScoreWeights) -> Tuple[float, dict[str, Any]]:
    """Length reward by categorical difficulty: easy / medium / hard (all lengths are token lengths)."""
    x = float(max(0, int(resp_token_len)))
    t = float(max(1, int(weights.medium_len)))

    sigma_easy_hard = max(1e-6, t / 3.0)
    sigma_medium = max(1e-6, t / 6.0)

    # easy: 2 * exp(-(x^2) / (2 * (t/3)^2))
    easy_reward = weights.len_mag * math.exp(-(x**2) / (2.0 * (sigma_easy_hard**2)))
    # medium: 2 * exp(-0.5 * ((x - t/2)/(t/6))^2)
    medium_reward = weights.len_mag * math.exp(-0.5 * (((x - t / 2.0) / sigma_medium) ** 2))
    # hard: 2 * exp(-((x - t)^2) / (2 * (t/3)^2))
    hard_reward = weights.len_mag * math.exp(-((x - t) ** 2) / (2.0 * (sigma_easy_hard**2)))

    if difficulty_level == "easy":
        r_len = easy_reward
        ideal_len = 0
        tolerance = sigma_easy_hard
    elif difficulty_level == "hard":
        r_len = hard_reward
        ideal_len = t
        tolerance = sigma_easy_hard
    else:
        r_len = medium_reward
        ideal_len = t / 2.0
        tolerance = sigma_medium

    return r_len, {
        "mode": difficulty_level,
        "ideal_token_len": int(ideal_len),
        "tolerance_token_len": int(max(1, round(tolerance))),
        "target_token_len": int(t),
        "x_token_len": int(x),
        "easy_score": float(easy_reward),
        "medium_score": float(medium_reward),
        "hard_score": float(hard_reward),
    }


def _call_llm(
    *,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_schema_name: Optional[str] = None,
    response_schema: Optional[dict[str, Any]] = None,
    extra_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Single LLM call entrypoint using OpenAI chat payload format."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload: dict[str, Any] = {"messages": messages}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if extra_context is not None:
        payload["extra_context"] = extra_context

    # Default path: call OpenAI-compatible API (vLLM/Qwen compatible).
    ctx = extra_context or {}
    vllm_cfg_raw = ctx.get("vllm_openai") or {}
    cfg = VLLMOpenAIConfig(
        base_url=vllm_cfg_raw.get(
            "base_url",
            os.getenv("VLLM_BASE_URL", os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode")),
        ),
        api_key=vllm_cfg_raw.get("api_key", "sk-6702e7de01c84cb88059105db0205e63"),
        model=vllm_cfg_raw.get("model", os.getenv("VLLM_MODEL", os.getenv("QWEN_MODEL", "qwen3-coder-plus"))),
        timeout_s=int(vllm_cfg_raw.get("timeout_s", 60)),
        max_tokens=int(vllm_cfg_raw.get("max_tokens", 2048)),
        temperature=float(vllm_cfg_raw.get("temperature", 0.0)),
        top_p=float(vllm_cfg_raw.get("top_p", 0.95)),
    )
    if not cfg.model:
        raise ValueError("Model is empty. Set VLLM_MODEL/QWEN_MODEL or llm_extra_context['vllm_openai']['model'].")

    request_payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": payload.get("temperature", cfg.temperature),
        "top_p": payload.get("top_p", cfg.top_p),
        "max_tokens": payload.get("max_tokens", cfg.max_tokens),
    }
    if response_schema is not None:
        if not response_schema_name:
            raise ValueError("response_schema_name is required when response_schema is set")
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_schema_name,
                "schema": response_schema,
            },
        }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    import requests

    base = cfg.base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"
    response = requests.post(
        url,
        headers=headers,
        json=request_payload,
        timeout=cfg.timeout_s,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, dict):
        parsed_content: Any = content
    elif isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
        parsed_content = content
    elif content is None:
        parsed_content = ""
    else:
        parsed_content = str(content).strip()
    reasoning = message.get("reasoning", message.get("reasoning_content"))
    return {"content": parsed_content, "reasoning": reasoning, "message": message}


def _llm_output_to_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict) and "content" in raw:
        return _llm_output_to_dict(raw["content"])
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("LLM response JSON must be an object")
    raise TypeError(f"Unsupported LLM response type: {type(raw).__name__}")


def compute_score(
    model_output: str,
    gold_label: int,
    resp_token_len: int,
    # required: pass as (yes_prob, no_prob), then we apply softmax and compute margin.
    yes_no_probs: Optional[Tuple[float, float]] = None,
    # llm_extra_context supports per-step params:
    # temperature/rubric_temperature/judge_temperature,
    # top_p/rubric_top_p/judge_top_p,
    # max_tokens/rubric_max_tokens/judge_max_tokens.
    reference_sql: Optional[str] = None,
    predicted_sql: Optional[str] = None,
    schema: Optional[str] = None,
    llm_extra_context: Optional[dict[str, Any]] = None,
    return_breakdown: bool = False,
) -> Any:

    # TODO:
    # ✅️ 先用DAtaSQL，infer一下把训练数据的SQL执行计算执行时间以及把超时报错的删了
    # ✅️ 完善代码的训练逻辑
    # [] 完善prompt。
    # [] 构建数据集
    # ✅️ 测出平局token 难度
    """
    difficulty              count          avg       median
    --------------------------------------------------------
    challenging               145     435.1517     408.0000
    moderate                  464     365.2047     357.0000
    simple                    925     310.7924     298.0000
    """

    """Compute reward = format + label(softmax) + llm + length.

    Priority:
    1) Format reward: original strict format check.
    2) Label reward: softmax margin over YES/NO probabilities.
    3) LLM reward: call OpenAI-compatible API, judge score in [0,10].
    4) Length reward (all lengths are token lengths):
       - easy: shorter is better (gaussian centered at 0)
       - medium: medium length is best (gaussian centered at medium_len / 2)
       - hard: longer is better (gaussian centered at medium_len)
       - length reward is enabled only when LLM reward equals 2.0

    Total score is clipped to [0, 10].
    """
    weights = ScoreWeights()
    resp_token_len = int(resp_token_len)
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
    if yes_no_probs is None:
        raise ValueError("yes_no_probs is required, expected (yes_prob, no_prob)")

    label_meta: dict[str, Any] = {"source": "yes_no_probs"}
    yes_prob = float(yes_no_probs[0])
    no_prob = float(yes_no_probs[1])
    p_yes, p_no = _softmax2(yes_prob, no_prob)
    margin = (p_yes - p_no) if gold_label == 1 else (p_no - p_yes)
    margin_pos = _clip(margin, 0.0, 1.0)
    r_label = weights.label_softmax_mag * margin_pos
    label_meta.update(
        {
            "yes_prob": yes_prob,
            "no_prob": no_prob,
            "p_yes": p_yes,
            "p_no": p_no,
            "margin": margin,
            "margin_pos": margin_pos,
        }
    )


    # -------------------------
    # 3) LLM score
    # -------------------------
    llm_result = {
        "rubric": None,
        "key_points": None,
        "score": None,
        "difficulty": None,
        "reason": None,
    }
    llm_meta: dict[str, Any] = {"enabled": False}

    missing_fields = []
    if reference_sql is None:
        missing_fields.append("reference_sql")
    if predicted_sql is None:
        missing_fields.append("predicted_sql")
    if schema is None:
        missing_fields.append("schema")
    if 0 < len(missing_fields) < 3:
        raise ValueError(f"Missing required fields for LLM judging: {', '.join(missing_fields)}")

    if not missing_fields:
        # Rubric and judge differ only in prompt and call params.
        llm_ctx = llm_extra_context or {}

        shared_temperature = llm_ctx.get("temperature")
        rubric_temperature = llm_ctx.get("rubric_temperature", shared_temperature)
        judge_temperature = llm_ctx.get("judge_temperature", shared_temperature)

        shared_top_p = llm_ctx.get("top_p")
        rubric_top_p = llm_ctx.get("rubric_top_p", shared_top_p)
        judge_top_p = llm_ctx.get("judge_top_p", shared_top_p)

        shared_max_tokens = llm_ctx.get("max_tokens")
        rubric_max_tokens = llm_ctx.get("rubric_max_tokens", shared_max_tokens)
        judge_max_tokens = llm_ctx.get("judge_max_tokens", shared_max_tokens)
        user_question = (
            llm_ctx.get("question")
            or llm_ctx.get("user_question")
            or llm_ctx.get("nl_question")
            or llm_ctx.get("query")
            or ""
        )

        rubric_user_prompt = f"""You are an expert NL2SQL rubric writer.
Generate a self-contained rubric for evaluating whether predicted_sql correctly answers the user's NL2SQL intent under this schema.
When writing scoring criteria, explicitly ground them in:
1) the reference sql (Reference_sql)
2) the user question (User_question).

## Model_output: 
{text}

## Reference_sql: 
{reference_sql}

## Predicted_sql: 
{predicted_sql}

## Schema: 
{schema}

## User_question: 
{user_question}

## NL2SQL rubric focus:
- Semantic correctness against the user intent (not only SQL syntax).
- Correct schema grounding: tables/columns exist and are used correctly; no hallucinated fields.
- Correct logic: filters, joins, aggregation, GROUP BY/HAVING, ORDER BY, LIMIT/TOP, DISTINCT.
- Correct value constraints: constants, ranges, date/time handling, units, inclusivity/exclusivity.
- Correct result shape: selected columns, granularity, ordering, and cardinality.
- Common pitfalls: missing key filters, wrong join keys causing duplication/drop, wrong aggregation level, unrelated row leakage.
- Consistency with verdict: include criteria connecting SQL quality and <answer>YES|NO when relevant.
- Keep criteria concise and self-contained; do not copy large raw blocks from inputs.

## Output format requirements:
1) Return a JSON object with exactly two top-level keys: rubric, key_points.
2) rubric must be a JSON array with 7-20 items (choose by complexity).
3) Each rubric item must contain exactly three keys: title, description, weight.
4) title must be 2-4 words.
5) description must be one sentence and start with exactly one prefix:
   - Essential Criteria:
   - Important Criteria:
   - Optional Criteria:
   - Pitfall Criteria: Does not ...
   - Pitfall Criteria: Recommends ...
6) Weight rules:
   - Essential/Important/Optional: integer 1-5 (Essential usually 5, Important usually 3-4, Optional usually 1-2).
   - Pitfall: -1 or -2.
7) No extra keys are allowed in each rubric item."""
        rubric_raw = _call_llm(
            user_prompt=rubric_user_prompt,
            system_prompt=_RUBRIC_SYSTEM_PROMPT,
            temperature=rubric_temperature,
            top_p=rubric_top_p,
            max_tokens=rubric_max_tokens,
            response_schema_name="sql_rubric_response",
            response_schema=_RubricResponse.model_json_schema(),
            extra_context=llm_extra_context,
        )
        rubric_result = _llm_output_to_dict(rubric_raw)
        rubric = rubric_result.get("rubric")
        key_points = rubric_result.get("key_points")

        judge_user_prompt = f"""gold_label: {gold_label}
user_question: {user_question}
model_output: {text}
reference_sql: {reference_sql}
predicted_sql: {predicted_sql}
schema: {schema}
rubric: {rubric}
key_points: {key_points}

You are an expert NL2SQL evaluator.
Given the user question and the model's SQL response, rate overall response quality from 1 to 10.
Use rubric and key_points as primary criteria, and use reference_sql as golden guidance (not necessarily exhaustive).
Focus on semantic correctness, schema grounding, SQL logic correctness, and result alignment with user intent.

Scoring anchors:
- 1-2: response is mostly incorrect or irrelevant to the NL2SQL task.
- 3-4: severe logical/schema errors; intent is largely unmet.
- 5-6: partially correct intent coverage with notable SQL mistakes.
- 7-8: mostly correct SQL with minor issues.
- 9-10: fully correct or near-perfect SQL aligned with intent and schema.

Determine difficulty based on question/SQL complexity:
- easy: simple single-table lookup/filter or straightforward aggregation.
- medium: moderate joins/aggregation/conditions.
- hard: complex multi-step logic, nested queries, advanced grouping/window logic, or tricky constraints.

Output format requirements:
1) Return a JSON object only (no markdown, no extra text).
2) JSON must contain exactly three keys: score, difficulty, reason.
3) score must be an integer in [1, 10].
4) difficulty must be one of: easy, medium, hard.
5) reason must be concise (1-3 sentences) and cite key evidence from rubric-based evaluation."""
        judge_raw = _call_llm(
            user_prompt=judge_user_prompt,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            temperature=judge_temperature,
            top_p=judge_top_p,
            max_tokens=judge_max_tokens,
            response_schema_name="sql_judge_response",
            response_schema=_JudgeResponse.model_json_schema(),
            extra_context=llm_extra_context,
        )
        judge_result = _llm_output_to_dict(judge_raw)

        llm_result = {
            "rubric": rubric,
            "key_points": key_points,
            "score": judge_result.get("score"),
            "difficulty": judge_result.get("difficulty"),
            "reason": judge_result.get("reason"),
        }
        llm_meta.update(
            {
                "enabled": True,
                "rubric_raw": rubric_result,
                "judge_raw": judge_result,
                "rubric_reasoning": rubric_raw.get("reasoning"),
                "judge_reasoning": judge_raw.get("reasoning"),
            }
        )

    llm_meta["result"] = llm_result

    llm_score_value = llm_result.get("score")
    llm_score_raw = float(llm_score_value) if llm_score_value is not None else None
    if llm_score_raw is None:
        r_llm = 0.0
        llm_score_10 = None
    else:
        llm_score_10 = _clip(llm_score_raw, 0.0, 10.0)
        llm_score_01 = llm_score_10 / 10.0
        r_llm = weights.llm_reward_mag * llm_score_01

    # -------------------------
    # 4) Length score by difficulty level
    # -------------------------
    llm_difficulty_value = llm_result.get("difficulty")
    if llm_difficulty_value is None:
        difficulty_level = "medium"
        difficulty_source = "default"
    else:
        difficulty_level = str(llm_difficulty_value).strip().lower()
        if difficulty_level not in _DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty '{llm_difficulty_value}', expected easy/medium/hard")
        difficulty_source = "llm"

    r_len_raw, len_meta = _compute_length_reward(resp_token_len=resp_token_len, difficulty_level=difficulty_level, weights=weights)
    len_reward_enabled = math.isclose(r_llm, 2.0, rel_tol=0.0, abs_tol=1e-8)
    r_len = r_len_raw if len_reward_enabled else 0.0
    len_meta["enabled"] = bool(len_reward_enabled)
    len_meta["raw_len_score"] = float(r_len_raw)
    len_meta["applied_len_score"] = float(r_len)
    len_meta["len_reward_gate_llm_score"] = 2.0
    ideal_token_len = int(len_meta["ideal_token_len"])
    tolerance_token_len = int(len_meta["tolerance_token_len"])
    length_mode = str(len_meta["mode"])

    total_raw = r_format + r_label + r_llm + r_len
    total = _clip(total_raw, 0.0, 10.0)

    summary = {
        "total_score": total,
        "total_score_raw": total_raw,
        "difficulty_level": difficulty_level,
        "judge_model_json": llm_result,
        "detail_score": {
            "r_format": r_format,
            "r_label": r_label,
            "r_llm": r_llm,
            "r_len": r_len,
        },
        "difficulty": {
            "level": difficulty_level,
            "source": difficulty_source,
            "ideal_token_len": ideal_token_len,
            "length_mode": length_mode,
        },
        "length_meta": len_meta,
        "llm_resp": processed,
        "resp_token_len": resp_token_len,
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
        "difficulty_level": difficulty_level,
        "judge_model_json": llm_result,
        "meta": {
            "strict_ok": bool(strict_ok),
            "pred": pred,
            "gold": gold,
            "token_len": int(resp_token_len),
            "header_error": err,
            "label_meta": label_meta,
            "llm_meta": llm_meta,
            "difficulty": {
                "level": difficulty_level,
                "source": difficulty_source,
                "ideal_token_len": int(ideal_token_len),
                "tolerance_token_len": int(tolerance_token_len),
                "length_mode": length_mode,
            },
            "length_meta": len_meta,
        },
    }


# -------------------------
# Quick self-checks (optional)
# -------------------------
if __name__ == "__main__":
    sample_question = """What is the highest eligible free rate for K-12 students in the schools in Alameda County?"""
    sample_schema = """-- free and reduced-price meals
CREATE TABLE `frpm` (
,    `cdscode` text PRIMARY KEY -- example: [01100170109835, 01100170112607]
    `academic year` text -- example: [2014-2015]
    `county code` text -- example: [01, 02]
    `district code` integer -- example: [10017, 31609]
    `school code` text -- example: [0109835, 0112607]
    `county name` text -- example: [Alameda, Alpine]
    `district name` text --
    `school name` text -- example: [FAME Public Charter]
    `district type` text -- example: [State Special Schools]
    `school type` text -- example: [K-12 Schools (Public), High Schools (Public)]
    `educational option type` text -- example: [Traditional, Juvenile Court School]
    `nslp provision status` text -- example: [Breakfast Provision 2, Provision 2]
    `charter school (y/n)` integer -- example: [1, 0]
    `charter school number` text -- example: [0728, 0811]
    `charter funding type` text -- example: [Directly funded, Locally funded]
    `irc` integer -- example: [1, 0]
    `low grade` text -- example: [K, 9]
    `high grade` text -- example: [12, 8]
    `enrollment (k-12)` real -- example: [1087.0, 395.0]
    `free meal count (k-12)` real -- example: [565.0, 186.0]
    `percent (%) eligible free (k-12)` real -- example: [0.519779208831647, 0.470886075949367]
    `frpm count (k-12)` real -- example: [715.0, 186.0]
    `percent (%) eligible frpm (k-12)` real -- example: [0.657773689052438, 0.470886075949367]
    `enrollment (ages 5-17)` real -- example: [1070.0, 376.0]
    `free meal count (ages 5-17)` real -- example: [553.0, 182.0]
    `percent (%) eligible free (ages 5-17)` real -- example: [0.516822429906542, 0.484042553191489]
    `frpm count (ages 5-17)` real -- example: [702.0, 182.0]
    `percent (%) eligible frpm (ages 5-17)` real -- example: [0.65607476635514, 0.484042553191489]
    `2013-14 calpads fall 1 certification status` integer -- example: [1]
    CONSTRAINT fk_frpm_cdscode FOREIGN KEY (`cdscode`) REFERENCES `schools` (`cdscode`)
);

CREATE TABLE `satscores` (
,    `cds` text PRIMARY KEY -- example: [10101080000000, 10101080109991]
    `rtype` text -- example: [D, S]
    `sname` text --  school name, example: [FAME Public Charter]
    `dname` text --  district name, example: [Alameda Unified]
    `cname` text --  county name, example: [Alameda, Amador]
    `enroll12` integer --  enrollment (1st-12nd grade), example: [398, 62]
    `numtsttakr` integer --  number of test takers, example: [88, 17]
    `avgscrread` integer --  average scores in reading, example: [418, 503]
    `avgscrmath` integer --  average scores in math, example: [418, 546]
    `avgscrwrite` integer --  average scores in writing, example: [417, 505]
    `numge1500` integer --  number of test takers whose total sat scores are greater or equal to 1500, example: [14, 9]
    CONSTRAINT fk_satscores_cds FOREIGN KEY (`cds`) REFERENCES `schools` (`cdscode`)
);

CREATE TABLE `schools` (
,    `cdscode` text PRIMARY KEY -- example: [01100170000000, 01100170109835]
    `ncesdist` text --  national center for educational statistics school district identification number, example: [0691051, 0600002]
    `ncesschool` text --  national center for educational statistics school identification number, example: [10546, 10947]
    `statustype` text -- example: [Active, Closed]
    `county` text -- example: [Alameda, Alpine]
    `district` text --
    `school` text -- example: [FAME Public Charter]
    `street` text -- example: [313 West Winton Avenue]
    `streetabr` text --  street address, example: [313 West Winton Ave.]
    `city` text -- example: [Hayward, Newark]
    `zip` text -- example: [94544-1136, 94560-5359]
    `state` text -- example: [CA]
    `mailstreet` text -- example: [313 West Winton Avenue]
    `mailstrabr` text --  mailing street address, example: [313 West Winton Ave.]
    `mailcity` text --  mailing city, example: [Hayward, Newark]
    `mailzip` text --  mailing zip, example: [94544-1136, 94560-5359]
    `mailstate` text --  mailing state, example: [CA]
    `phone` text -- example: [(510) 887-0152, (510) 596-8901]
    `ext` text --  extension, example: [130, 1240]
    `website` text -- example: [www.acoe.org, www.envisionacademy.org/]
    `opendate` date -- example: [2005-08-29, 2006-08-28]
    `closeddate` date -- example: [2015-07-31, 2015-06-30]
    `charter` integer -- example: [1, 0]
    `charternum` text -- example: [0728, 0811]
    `fundingtype` text -- example: [Directly funded, Locally funded]
    `doc` text --  district ownership code, example: [00, 31]
    `doctype` text --  the district ownership code type, example: [State Special Schools]
    `soc` text --  school ownership code, example: [65, 66]
    `soctype` text --  school ownership code type, example: [K-12 Schools (Public), High Schools (Public)]
    `edopscode` text --  education option code, example: [TRAD, JUV]
    `edopsname` text --  educational option name, example: [Traditional, Juvenile Court School]
    `eilcode` text --  educational instruction level code, example: [ELEMHIGH, HS]
    `eilname` text --  educational instruction level name, example: [High School]
    `gsoffered` text --  grade span offered, example: [K-12, 9-12]
    `gsserved` text --  grade span served., example: [K-12, 9-12]
    `virtual` text -- example: [P, N]
    `magnet` integer -- example: [0, 1]
    `latitude` real -- example: [37.658212, 37.521436]
    `longitude` real -- example: [-122.09713, -121.99391]
    `admfname1` text --  administrator's first name 1, example: [L Karen, Laura]
    `admlname1` text --  administrator's last name 1, example: [Monroe, Robell]
    `admemail1` text --  administrator's email address 1, example: [lkmonroe@acoe.org, laura@envisionacademy.org]
    `admfname2` text --  administrator's first name 2, example: [Sau-Lim (Lance), Jennifer]
    `admlname2` text --  administrator's last name 2, example: [Tsang, Koelling]
    `admemail2` text --  administrator's email address 2, example: [stsang@unityhigh.org, jkoelling@efcps.net]
    `admfname3` text --  administrator's first name 3, example: [Drew, Irma]
    `admlname3` text --  administrator's last name 3, example: [Sarratore, Munoz]
    `admemail3` text --  administrator's email address 3, example: [gmunoz@piedmont.k12.ca.us]
    `lastupdate` date -- example: [2015-06-23, 2015-09-01]
);"""
    sample_out = """<think> We need judge if model SQL answers the question: "highest eligible free rate for K-12 students in the schools in Alameda County". In schema, frpm has column `percent (%) eligible free (k-12)` and `county name`. Model query selects MAX of that column from frpm where county name='Alameda'. That matches requirement: highest eligible free rate among schools in Alameda County for K-12. No need join. So it completes. Minor: county name examples use 'Alameda' so filter ok. Therefore YES. </think> <answer>YES</answer>"""
    sample_answer_sql = """SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1"""
    provided_case = compute_score(
        sample_out,
        gold_label=1,
        resp_token_len=len(sample_out),
        yes_no_probs=(3.2, 0.3),
        reference_sql=sample_answer_sql,
        predicted_sql=sample_answer_sql,
        schema=sample_schema,
        return_breakdown=True,
    )
    print("provided_case_question:", sample_question)
    print("final_total_score:", provided_case["total"])
    print("difficulty_level:", provided_case["difficulty_level"])
    print("judge_model_json:", json.dumps(provided_case["judge_model_json"], ensure_ascii=False))
    print(
        "final_detail_scores:",
        {
            "format": provided_case["format"],
            "label": provided_case["label"],
            "llm": provided_case["llm"],
            "length": provided_case["length"],
        },
    )
    print(len(sample_out))
    # print(json.dumps(provided_case, ensure_ascii=False, indent=2))
