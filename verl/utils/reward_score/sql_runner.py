# sql_runner.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sqlite3
import sys


def run_sql(db_path: str, sql: str):
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"Database file not found: {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()

        # JSON 里放 list[list]，主进程再转 tuple 用于 set
        return {"ok": True, "rows": [list(r) for r in rows]}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--sql", required=True)
    args = ap.parse_args()

    payload = run_sql(args.db, args.sql)

    # 强制只输出一行 JSON，方便主进程读取
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()

    # 用退出码表达成功/失败
    sys.exit(0 if payload.get("ok") else 2)


if __name__ == "__main__":
    main()