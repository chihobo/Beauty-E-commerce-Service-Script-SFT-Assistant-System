# -*- coding: utf-8 -*-
"""T6 优质筛选测试（对应 T06、T07、T08 / R4）。"""
import pytest

from src.data_pipeline.filter_quality import (
    base_gate,
    filter_sessions,
    llm_judge,
    session_to_text,
)
from tests.stub_client import StubClient


def _msg(sender, text):
    return {"timestamp": 0, "sender": sender, "text": text}


def _good_session(sid="s1", rating=None, ordered=False, turns=3):
    """构造一个能过基础门槛的会话：3 轮对话、结尾非抱怨、客服非敷衍。"""
    msgs = []
    for i in range(turns):
        msgs.append(_msg("customer", f"这款适合我吗，第{i + 1}问"))
        msgs.append(_msg("service", "宝子，这款很清爽适合你的哦，放心拍"))
    return {
        "session_id": sid,
        "customer_id": "c1",
        "rating": rating,
        "ordered_within_24h": ordered,
        "messages": msgs,
    }


class _NeverCalledClient(StubClient):
    """被调用即抛错的桩：验证快速通道/成交通道不触发 LLM 判别。"""

    def __init__(self):
        super().__init__(self._boom)

    def _boom(self, prompt):
        raise AssertionError("不该调用 LLM 判别")


# ---------- 基础门槛（T06） ----------

def test_base_gate_rejects_short_session():
    s = _good_session(turns=1)
    ok, reason = base_gate(s)
    assert not ok
    assert "3轮" in reason


def test_base_gate_rejects_ending_complaint():
    s = _good_session()
    s["messages"].append(_msg("customer", "什么垃圾店，我再也不来了"))
    ok, reason = base_gate(s)
    assert not ok
    assert "抱怨" in reason


def test_base_gate_rejects_perfunctory_service():
    msgs = [
        _msg("customer", "在吗"),
        _msg("service", "在"),
        _msg("customer", "好用吗"),
        _msg("service", "嗯"),
        _msg("customer", "多少钱"),
        _msg("service", "哦"),
    ]
    s = _good_session()
    s["messages"] = msgs
    ok, reason = base_gate(s)
    assert not ok
    assert "敷衍" in reason


def test_base_gate_passes_good_session():
    ok, _ = base_gate(_good_session())
    assert ok


# ---------- 快速通道（T07） ----------

def test_fast_track_high_rating_selected():
    passed, report = filter_sessions([_good_session(rating=5)], _NeverCalledClient())
    assert len(passed) == 1
    assert report["fast_selected"] == 1


def test_fast_track_low_rating_rejected():
    passed, report = filter_sessions([_good_session(rating=2)], _NeverCalledClient())
    assert len(passed) == 0
    assert report["fast_rejected"] == 1


# ---------- 多信号通道（T08） ----------

def test_ordered_signal_selected_without_llm():
    passed, report = filter_sessions([_good_session(ordered=True)], _NeverCalledClient())
    assert len(passed) == 1
    assert report["multi_selected"] == 1


def test_llm_judge_yes_selected():
    passed, report = filter_sessions([_good_session()], StubClient("yes"))
    assert len(passed) == 1
    assert report["multi_selected"] == 1


def test_llm_judge_no_rejected():
    passed, report = filter_sessions([_good_session()], StubClient("no"))
    assert len(passed) == 0
    assert report["multi_rejected"] == 1


def test_llm_judge_unparsable_raises():
    """LLM 返回非 yes/no 时直接报错，不做兜底。"""
    with pytest.raises(RuntimeError):
        llm_judge("客户：在吗\n客服：在的", StubClient("好的"))


def test_funnel_report_closed():
    """漏斗报告各级数量闭合：总数 = 过门槛 + 门槛淘汰；入选 = 快速 + 多信号。"""
    sessions = [
        _good_session("s1", rating=5),          # 快速通道入选
        _good_session("s2", rating=1),          # 快速通道淘汰
        _good_session("s3", ordered=True),      # 成交信号入选
        _good_session("s4"),                    # LLM 判别（yes）入选
        _good_session("s5", turns=1),           # 基础门槛淘汰（不足3轮）
    ]
    passed, report = filter_sessions(sessions, StubClient("yes"))
    assert report["total"] == 5
    assert report["total"] == report["base_passed"] + report["base_rejected"]
    assert report["base_passed"] == (
        report["fast_selected"] + report["fast_rejected"]
        + report["multi_selected"] + report["multi_rejected"]
    )
    assert report["final_selected"] == report["fast_selected"] + report["multi_selected"]
    assert len(passed) == report["final_selected"] == 3


def test_session_to_text_format():
    text = session_to_text(_good_session(turns=1))
    assert "客户：" in text
    assert "客服：" in text
