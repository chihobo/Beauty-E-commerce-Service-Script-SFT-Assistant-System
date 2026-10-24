# -*- coding: utf-8 -*-
"""T7 分类与配比测试（对应 T09 / R5）。"""
import pytest

from src.data_pipeline.label_category import classify_session, rebalance
from tests.stub_client import StubClient


def _session(sid, category=None):
    s = {
        "session_id": sid,
        "customer_id": "c1",
        "rating": 5,
        "ordered_within_24h": False,
        "messages": [{"timestamp": 0, "sender": "customer", "text": "在吗"}],
    }
    if category is not None:
        s["category"] = category
    return s


def test_classify_returns_valid_category():
    assert classify_session("客户：多少钱\n客服：129元", StubClient("售前")) == "售前"
    assert classify_session("客户：快递呢\n客服：在转运", StubClient("售后")) == "售后"
    assert classify_session("客户：好累\n客服：抱抱", StubClient("闲聊安抚")) == "闲聊安抚"


def test_classify_tolerates_extra_whitespace():
    assert classify_session("...", StubClient("  售前 \n")) == "售前"


def test_classify_invalid_response_raises():
    """LLM 返回无法识别的类别时直接报错，不做兜底。"""
    with pytest.raises(RuntimeError):
        classify_session("客户：在吗", StubClient("不知道"))


def test_rebalance_presale_downsampled():
    """售前降采样：售前 :（售后+闲聊） = 60 : 40。"""
    sessions = (
        [_session(f"pre{i}", "售前") for i in range(85)]
        + [_session(f"after{i}", "售后") for i in range(12)]
        + [_session(f"chat{i}", "闲聊安抚") for i in range(4)]
    )
    result = rebalance(sessions)
    pre = [s for s in result if s["category"] == "售前"]
    after = [s for s in result if s["category"] == "售后"]
    chat = [s for s in result if s["category"] == "闲聊安抚"]
    # 售前被降到 24（16 * 60/40），售后与闲聊全保留
    assert len(pre) == 24
    assert len(after) == 12
    assert len(chat) == 4
    # 降采样是原子集，不放回
    assert len({s["session_id"] for s in pre}) == 24


def test_rebalance_no_downsample_when_presale_short():
    """售前不足目标量时不做降采样，全量保留。"""
    sessions = (
        [_session(f"pre{i}", "售前") for i in range(10)]
        + [_session(f"after{i}", "售后") for i in range(12)]
        + [_session(f"chat{i}", "闲聊安抚") for i in range(4)]
    )
    result = rebalance(sessions)
    assert len([s for s in result if s["category"] == "售前"]) == 10
    assert len(result) == 26
