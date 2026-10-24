# -*- coding: utf-8 -*-
"""T5 会话拆分：按 30 分钟间隔把消息流水拆成会话（对应 R3）。

规则：同一客户按时间排序，相邻两条消息间隔超过 SESSION_GAP_MINUTES（30 分钟）
即拆为两个会话。session_id 重建规则与 T3 生成侧一致：
"{customer_id}#{序号:02d}"（序号为该客户会话按时间先后的编号），
用于把 sessions_meta.json 中的打星/成交标记挂回会话。

入口：python -m src.data_pipeline.split_sessions
"""
import json

from src.config import DATA_PROCESSED, DATA_RAW, SESSION_GAP_MINUTES


def split_customer_messages(msgs, gap_minutes=SESSION_GAP_MINUTES):
    """把同一客户按时间排序的消息列表拆成若干会话（参考项目叙事中的 split_sessions 改写）。"""
    sessions = []
    current = [msgs[0]]
    for i in range(1, len(msgs)):
        gap = (msgs[i]["timestamp"] - msgs[i - 1]["timestamp"]) / 60
        if gap > gap_minutes:
            sessions.append(current)
            current = [msgs[i]]
        else:
            current.append(msgs[i])
    sessions.append(current)
    return sessions


def build_sessions(messages, sessions_meta=None, gap_minutes=SESSION_GAP_MINUTES):
    """把全量消息流水拆成会话列表，并挂回会话级元信息。

    返回 [{session_id, customer_id, rating, ordered_within_24h, messages}]，
    messages 元素为 {timestamp, sender, text}。
    """
    by_customer = {}
    for m in messages:
        by_customer.setdefault(m["customer_id"], []).append(m)
    sessions = []
    for customer_id, msgs in by_customer.items():
        msgs.sort(key=lambda m: m["timestamp"])
        for idx, sess_msgs in enumerate(split_customer_messages(msgs, gap_minutes)):
            session_id = f"{customer_id}#{idx:02d}"
            info = (sessions_meta or {}).get(session_id, {})
            sessions.append({
                "session_id": session_id,
                "customer_id": customer_id,
                "rating": info.get("rating"),
                "ordered_within_24h": info.get("ordered_within_24h", False),
                "messages": [
                    {"timestamp": m["timestamp"], "sender": m["sender"], "text": m["text"]}
                    for m in sess_msgs
                ],
            })
    sessions.sort(key=lambda s: s["messages"][0]["timestamp"])
    return sessions


def main():
    src = DATA_PROCESSED / "messages_desensitized.jsonl"
    meta_src = DATA_RAW / "sessions_meta.json"
    dst = DATA_PROCESSED / "sessions.jsonl"

    with open(src, encoding="utf-8") as f:
        messages = [json.loads(line) for line in f if line.strip()]
    with open(meta_src, encoding="utf-8") as f:
        sessions_meta = json.load(f)

    sessions = build_sessions(messages, sessions_meta)
    rated = sum(1 for s in sessions if s["rating"] is not None)

    with open(dst, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[T5] 会话拆分完成: {len(messages)} 条消息 -> {len(sessions)} 个会话"
          f"（间隔>{SESSION_GAP_MINUTES}分钟拆开），带评分 {rated} 个")
    print(f"  输出: {dst}")


if __name__ == "__main__":
    main()
