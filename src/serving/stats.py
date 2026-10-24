# -*- coding: utf-8 -*-
"""使用统计（R14 看板）：草稿采纳率等统计的纯函数实现。

从 Streamlit 界面抽出的纯逻辑，便于脱离 UI 单测：
- load_jsonl：读取 jsonl 日志（决策日志 / 采纳记录）；
- calc_adoption_rate：按采纳记录统计直接发送 / 修改后发送 / 弃用占比与采纳率。
采纳口径：直接发送 + 修改后发送 视为采纳，弃用视为未采纳。
"""
import json
from pathlib import Path

# 草稿处理的三种人工动作（写入采纳记录 adoption_log.jsonl 的 action 字段）
ADOPTION_ACTIONS = ("直接发送", "修改后发送", "弃用")

# 视为采纳的动作
_ADOPTED_ACTIONS = ("直接发送", "修改后发送")


def load_jsonl(path) -> list[dict]:
    """读取 jsonl 文件为记录列表；文件不存在返回空列表，空行跳过。"""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def calc_adoption_rate(records: list[dict]) -> dict:
    """统计草稿采纳情况。

    入参为采纳记录列表（每条含 action 字段，取值为 ADOPTION_ACTIONS 之一）。
    返回：
    - total：记录总数；
    - counts：三种动作各自条数；
    - adoption_rate：采纳率 = (直接发送 + 修改后发送) / 总数，无记录时为 0.0。
    """
    counts = {action: 0 for action in ADOPTION_ACTIONS}
    for record in records:
        action = record.get("action")
        if action in counts:
            counts[action] += 1
    total = len(records)
    adopted = sum(counts[a] for a in _ADOPTED_ACTIONS)
    return {
        "total": total,
        "counts": counts,
        "adoption_rate": adopted / total if total else 0.0,
    }
