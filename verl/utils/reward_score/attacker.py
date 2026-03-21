import os
import re
import json
import time
import logging
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from logging.handlers import RotatingFileHandler

# =========================
# Patterns (same as yours)
# =========================

_STRICT_PATTERN = re.compile(
    r"""(?isx)
    ^\s*
    <think>(?P<think>[\s\S]*?)</think>\s*
    <answer>[\s\S]*?```sql\s*(?P<sql>[\s\S]*?)\s*```\s*[\s\S]*?</answer>
    \s*$
    """,
)
_ANSWER_TAG = re.compile(r"<answer>\s*(?P<ans>[\s\S]*?)\s*</answer>", flags=re.IGNORECASE)
_THINK_TAG = re.compile(r"<think>\s*(?P<think>[\s\S]*?)\s*</think>", flags=re.IGNORECASE)

# Parse SQL fenced code
_SQL_FENCE = re.compile(r"```sql\s*([\s\S]*?)```", flags=re.IGNORECASE)

def _split_assistant(solution_str: str) -> Tuple[Optional[str], str]:
    if "Assistant:" in solution_str:
        return solution_str.split("Assistant:", 1)[1], ""
    if "<|im_start|>assistant" in solution_str:
        return solution_str.split("<|im_start|>assistant", 1)[1], ""
    return solution_str, ""

def _normalize_answer(raw: str) -> Optional[str]:
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
    m_think = _THINK_TAG.search(text)
    think_text = m_think.group("think") if m_think else None

    m_ans = _ANSWER_TAG.search(text)
    pred = _normalize_answer(m_ans.group("ans")) if m_ans else None
    return think_text, pred

def extract_sql_from_model_output(model_output: str) -> str:
    matches = _SQL_FENCE.findall(model_output)  # 返回所有 sql fence 内部内容（捕获组）
    if matches:
        return matches[-1].strip()
    return ""

# =========================
# Weights (same as yours)
# =========================
@dataclass
class ScoreWeights:
    strict_format: float = 0.5
    has_answer_tag_only: float = 0.1  # (你当前没用到，保留)

    label_correct_strict: float = 2.0
    label_correct_loose: float = 0.4

    len_mag: float = 0.5
    no_pred_len_penalty_mag: float = 0.25

    free_len: int = 512
    over_len_penalty_mag: float = 0.5

# =========================
# Logger (same style as yours)
# =========================
def get_score_logger(log_file: str = "logs/rollouts.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("score_logger")
    logger.setLevel(level)
    logger.propagate = False

    abs_path = os.path.abspath(log_file)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path:
            return logger

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    fh = RotatingFileHandler(
        abs_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | pid=%(process)d | %(message)s"))
    logger.addHandler(fh)
    return logger

_SCORE_LOGGER = get_score_logger("logs/rollouts_atk.log")

# =========================
# Judge prompt (yours)
# =========================
sql_check_prompt = """Task Overview:
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
- Even small mismatches should be considered incorrect (e.g., extra columns, missing filters, wrong table joins, wrong ordering, wrong limit, wrong logic, etc.).
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

def render_sql_check_prompt(schema: str, question: str, attacker_sql: str) -> str:
    return (sql_check_prompt
            .replace("{{SCHEMA}}", schema or "")
            .replace("{{QUESTION}}", question or "")
            .replace("{{FILTER_SQL}}", attacker_sql or ""))

# =========================
# vLLM (OpenAI-compatible) client
# =========================
@dataclass
class VLLMJudgeConfig:
    base_url: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
    api_key: Optional[str] = os.getenv("VLLM_API_KEY", None)
    model: str = os.getenv("VLLM_MODEL", "/root/autodl-fs/huggingface/hub/models--Chinastark--diss/snapshots/450e2a760907371dd7b77286b3337a18ea20ecdc")
    timeout_s: int = 60
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.95
    # retry
    max_retries: int = 3
    retry_backoff_s: float = 0.8

def call_vllm_judge(prompt: str, cfg: VLLMJudgeConfig) -> str:
    """
    调用 vLLM 的 /v1/chat/completions（OpenAI-compatible）。
    你只要保证 vLLM 服务开启了 openai api server。
    """
    url = cfg.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=cfg.timeout_s,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            data = r.json()
            # OpenAI chat format:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt + 1 < cfg.max_retries:
                time.sleep(cfg.retry_backoff_s * (2 ** attempt))
            else:
                raise

    raise RuntimeError(f"vLLM judge call failed: {last_err}")

# ========================
# 执行sql 获取 label score
# ========================
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple


def run_sql_in_subprocess(
    python_exe: str,
    runner_py: str,
    db_path: str,
    sql: str,
    timeout: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """运行子进程并返回 (payload, error_msg). 超时/崩溃会返回 None + err."""

    # 不要 shell=True（更安全也更好跨平台）
    cmd = [python_exe, "-u", runner_py, "--db", db_path, "--sql", sql]

    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timed out after {timeout}s"

    # 子进程可能把错误写 stderr，也可能 stdout 不是合法 JSON
    out = (completed.stdout or "").strip()
    err = (completed.stderr or "").strip()

    if not out:
        return None, f"No stdout. stderr={err[:500]}"

    try:
        payload = json.loads(out)
    except Exception:
        return None, f"Invalid JSON from child. stdout={out[:500]} stderr={err[:500]}"

    if not payload.get("ok"):
        return payload, payload.get("error") or err or "Child reported failure"

    return payload, None


def evaluate_sql_match(attacker_sql: str, gt_sql: str, db_id: str, db_base_path: str, timeout: float = 30.0):
    attacker_sql = attacker_sql.strip()
    gt_sql = gt_sql.strip()

    db_path = os.path.join(db_base_path, db_id, f"{db_id}.sqlite")

    # 让 python_exe 指向当前解释器，避免环境不一致
    python_exe = sys.executable

    # runner 脚本路径：你可以改成绝对路径
    runner_py = os.path.join(os.path.dirname(__file__), "sql_runner.py")

    # --- 并行跑：用两个 subprocess 同时起，然后分别等待（各自 timeout） ---
    # 如果你更在意“总超时”，可以把 timeout 分摊或者用 Popen + 统一计时。

    payload_a, err_a = run_sql_in_subprocess(python_exe, runner_py, db_path, attacker_sql, timeout)
    payload_g, err_g = run_sql_in_subprocess(python_exe, runner_py, db_path, gt_sql, timeout)

    if err_a or err_g:
        print(f"❌ attacker err: {err_a}")
        print(f"❌ gt err:       {err_g}")
        print("Scoring: False (Execution Failed)")
        return False

    rows_a = payload_a["rows"]
    rows_g = payload_g["rows"]

    # set 比较（忽略顺序）
    try:
        attacker_set = set(tuple(r) for r in rows_a)
        gt_set = set(tuple(r) for r in rows_g)
    except TypeError:
        # 极少数情况下包含不可 hash 的对象：做字符串化兜底
        attacker_set = set(map(str, rows_a))
        gt_set = set(map(str, rows_g))

    is_match = (attacker_set == gt_set)

    print(f"Attacker Result: {attacker_set}")
    print(f"GT Result:       {gt_set}")
    print(f"Match:           {is_match}")

    # 你原来的逻辑：不匹配返回 True（像是在判断 attack 是否成功）
    # 如果你想要“SQL 等价则 True”，把这里改成：return is_match
    if not is_match:
        return True

    return False
# =========================
# ✅ LLM-as-Judge compute_score (the one you want)
# =========================
def compute_score(
    model_output: str,
    schema: str,
    question: str,
    resp_len: int,
    ground_truth: str,
    db_id: str,
    L: int = 2048,
    weights: ScoreWeights = ScoreWeights(),
    judge_cfg: VLLMJudgeConfig = VLLMJudgeConfig(),
    return_breakdown: bool = False,
) -> Any:
    """
    model_output: attacker 生成的内容（其中 SQL 在 ```sql ...``` 里）
    gold_label: 1->YES, 0->NO（表示 attacker SQL 是否真的能正确回答 question）
    schema/question: 用于 sql_check_prompt
    """
    # get the attacked sql
    attacker_sql = extract_sql_from_model_output(model_output)
    if attacker_sql == "" : return -0.5
    # judge the sql
    prompt = render_sql_check_prompt(schema=schema, question=question, attacker_sql=attacker_sql)
    judge_text = call_vllm_judge(prompt, judge_cfg)
    processed_jg, err = _split_assistant(judge_text)
    text_jg = processed_jg if processed_jg is not None else judge_text
    
    _, gold = _extract_loose(text_jg)
    
    processed, err = _split_assistant(model_output)
    text = processed if processed is not None else model_output
    # Strict match
    
    m_strict = _STRICT_PATTERN.match(text)
    strict_ok = m_strict is not None

    # Loose extraction (independent of strict)
    # think_text, pred = _extract_loose(text)

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
    # attacker_sql = attacker_sql.strip()
    # gt_sql = ground_truth
    is_ok = False
    if gold is not None and gold == "NO":
        r_label = 0.0
        correct = False
    else:
        is_ok = evaluate_sql_match(attacker_sql, ground_truth, db_id, db_base_path="/root/autodl-fs/train_databases", timeout=30)
        _SCORE_LOGGER.info(f"the sql is {is_ok}")
        if is_ok == True and strict_ok:
            r_label = weights.label_correct_strict
        elif is_ok == True:
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
    

    # 记录：输入 SQL + judge 回复 + 分数
    summary = {
        "total_score": total,
        "detail_score": {
            "r_format": r_format,
            "r_label": r_label ,
            "r_len": r_len,
        },
        "attacker_resp": model_output,
        "is_ok": is_ok,
        "gold": gold,
        "attacker_sql": attacker_sql,
        "judge_resp": judge_text,
        "len": resp_len
    }
    _SCORE_LOGGER.info(json.dumps(summary, ensure_ascii=False))
    return float(total)


# -------------------------
# Quick self-check (optional)
# -------------------------
if __name__ == "__main__":
    fake_attacker = "Assistant: 这里是输出\n```sql\nSELECT 1;\n```"
    # 下面这行会真的打到你的 vLLM judge 上
    # print(compute_score(fake_attacker, gold_label=0, schema="...", question="...", return_breakdown=True))
