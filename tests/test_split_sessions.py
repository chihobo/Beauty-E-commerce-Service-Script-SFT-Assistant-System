# -*- coding: utf-8 -*-
"""T5 会话拆分测试（对应 T04、T05 / R3）。"""
from src.data_pipeline.split_sessions import build_sessions, split_customer_messages

BASE_TS = 1_700_000_000  # 固定基准时间戳（秒）


def _msg(customer_id, offset_minutes, text="你好", sender="customer"):
    return {
        "timestamp": BASE_TS + offset_minutes * 60,
        "customer_id": customer_id,
        "nickname": "测试买家",
        "sender": sender,
        "text": text,
    }


def test_gap_over_30min_splits():
    """同一客户相邻消息间隔超过 30 分钟拆为两个会话（T04）。"""
    msgs = [
        _msg("c1", 0, "在吗"),
        _msg("c1", 2, "在的", "service"),
        _msg("c1", 33, "再问一下"),  # 距上一条 31 分钟
        _msg("c1", 35, "你说", "service"),
    ]
    sessions = split_customer_messages(msgs)
    assert len(sessions) == 2
    assert len(sessions[0]) == 2
    assert len(sessions[1]) == 2


def test_gap_within_30min_merges():
    """间隔 30 分钟内保持同一会话（T05）。"""
    msgs = [
        _msg("c1", 0, "在吗"),
        _msg("c1", 5, "在的", "service"),
        _msg("c1", 10, "多少钱"),
        _msg("c1", 15, "129元", "service"),
    ]
    sessions = split_customer_messages(msgs)
    assert len(sessions) == 1
    assert len(sessions[0]) == 4


def test_customers_not_mixed():
    """不同客户的消息交错时不串会话。"""
    msgs = [
        _msg("c1", 0, "c1第一条"),
        _msg("c2", 1, "c2第一条"),
        _msg("c1", 2, "c1第二条", "service"),
        _msg("c2", 3, "c2第二条", "service"),
        _msg("c1", 60, "c1新会话"),  # c1 间隔 58 分钟
        _msg("c2", 20, "c2继续"),    # c2 间隔 17 分钟
    ]
    sessions = build_sessions(msgs)
    assert len(sessions) == 3
    by_customer = {}
    for s in sessions:
        by_customer.setdefault(s["customer_id"], []).append(s)
    assert len(by_customer["c1"]) == 2
    assert len(by_customer["c2"]) == 1
    for s in sessions:
        for m in s["messages"]:
            # 会话内所有消息必须属于同一客户
            assert s["session_id"].startswith(s["customer_id"])


def test_meta_attached_by_session_id():
    """会话级元信息（rating / ordered_within_24h）按 session_id 挂回。"""
    msgs = [
        _msg("c1", 0, "第一条"),
        _msg("c1", 2, "回复", "service"),
        _msg("c1", 90, "新会话"),
        _msg("c1", 92, "回复", "service"),
    ]
    meta = {
        "c1#00": {"rating": 5, "ordered_within_24h": False},
        "c1#01": {"rating": None, "ordered_within_24h": True},
    }
    sessions = build_sessions(msgs, sessions_meta=meta)
    assert len(sessions) == 2
    first = next(s for s in sessions if s["session_id"] == "c1#00")
    second = next(s for s in sessions if s["session_id"] == "c1#01")
    assert first["rating"] == 5 and first["ordered_within_24h"] is False
    assert second["rating"] is None and second["ordered_within_24h"] is True
