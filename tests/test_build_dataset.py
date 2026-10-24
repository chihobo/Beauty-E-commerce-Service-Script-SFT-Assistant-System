# -*- coding: utf-8 -*-
"""T8 占位符化与训练集构建测试（对应 T10、T11 / R6、R7）。"""
from collections import Counter

from src.data_pipeline.build_dataset import (
    placeholderize,
    stratified_split,
    to_sft_sample,
)


def _msg(sender, text):
    return {"timestamp": 0, "sender": sender, "text": text}


def _session(sid, category="售前"):
    return {
        "session_id": sid,
        "customer_id": "c1",
        "category": category,
        "messages": [
            _msg("customer", "这款多少钱呀"),
            _msg("service", "宝子，现在活动价129元，还能领满99减10的券哦"),
            _msg("customer", "库存还多吗"),
            _msg("service", "宝子，库存50件，喜欢尽快拍哦"),
        ],
    }


# ---------- 占位符化（T10） ----------

def test_price_masked():
    assert placeholderize("现在活动价129元哦") == "现在活动价[当前活动价]哦"
    assert placeholderize("只要129.5元") == "只要[当前活动价]"


def test_discount_masked():
    assert placeholderize("领满99减10的券") == "领[当前优惠]的券"


def test_stock_masked():
    assert placeholderize("库存50件哦") == "库存[当前库存]哦"


def test_customer_text_not_masked():
    """只处理客服回复，客户消息中的数字不动。"""
    sample = to_sft_sample(_session("s1"))
    user_texts = [m["content"] for m in sample["messages"] if m["role"] == "user"]
    assert user_texts[0] == "这款多少钱呀"


def test_service_text_masked_in_sample():
    sample = to_sft_sample(_session("s1"))
    assistant = [m["content"] for m in sample["messages"] if m["role"] == "assistant"]
    assert "[当前活动价]" in assistant[0] and "129元" not in assistant[0]
    assert "[当前优惠]" in assistant[0]
    assert "库存[当前库存]" in assistant[1]


# ---------- messages 格式（T11） ----------

def test_sft_messages_format():
    sample = to_sft_sample(_session("s1"))
    msgs = sample["messages"]
    assert sample["category"] == "售前"
    assert msgs[0]["role"] == "system"
    assert "宝子" in msgs[0]["content"]
    roles = [m["role"] for m in msgs[1:]]
    assert roles == ["user", "assistant", "user", "assistant"]


# ---------- 分层抽样划分（T11） ----------

def _make_samples():
    samples = []
    for cat, n in (("售前", 60), ("售后", 30), ("闲聊安抚", 10)):
        for i in range(n):
            samples.append({"id": f"{cat}-{i}", "category": cat, "messages": []})
    return samples


def test_stratified_split_ratio_and_no_overlap():
    samples = _make_samples()
    train, test = stratified_split(samples, train_size=80, test_size=20, seed=42)
    assert len(train) == 80
    assert len(test) == 20
    # 无交集
    train_ids = {s["id"] for s in train}
    test_ids = {s["id"] for s in test}
    assert not (train_ids & test_ids)
    # 训练/测试类别分布一致（60/30/10）
    for part in (train, test):
        dist = Counter(s["category"] for s in part)
        total = len(part)
        assert abs(dist["售前"] / total - 0.6) < 0.05
        assert abs(dist["售后"] / total - 0.3) < 0.05
        assert abs(dist["闲聊安抚"] / total - 0.1) < 0.05


def test_stratified_split_scales_down_when_insufficient():
    """数据不够 800+120 时按比例缩放，全部数据仍被用完。"""
    samples = _make_samples()[:50]
    train, test = stratified_split(samples, train_size=800, test_size=120, seed=42)
    assert len(train) + len(test) == 50
    assert len(train) > len(test)


def test_split_reproducible():
    samples = _make_samples()
    t1, s1 = stratified_split(samples, train_size=80, test_size=20, seed=42)
    t2, s2 = stratified_split(samples, train_size=80, test_size=20, seed=42)
    assert [x["id"] for x in t1] == [x["id"] for x in t2]
    assert [x["id"] for x in s1] == [x["id"] for x in s2]
