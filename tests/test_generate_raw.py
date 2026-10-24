# -*- coding: utf-8 -*-
"""T3 数据生成测试（对应 T01 / R1）。"""
from collections import Counter

from src.data_pipeline import generate_raw
from src.data_pipeline.desensitize import scan_sensitive


def test_message_fields_complete():
    """消息流水字段齐全：timestamp/customer_id/nickname/sender/text。"""
    messages, _ = generate_raw.generate_corpus(num_customers=30, seed=42)
    assert len(messages) > 0
    for m in messages:
        assert isinstance(m["timestamp"], int)
        assert m["customer_id"]
        assert m["nickname"]
        assert m["sender"] in ("customer", "service")
        assert m["text"]


def test_rating_coverage_and_j_shape():
    """打星覆盖率约 30%（25%~35%），且呈 J 型分布（5 星为主、1 星次之）。"""
    _, meta = generate_raw.generate_corpus(seed=42)
    ratings = [v["rating"] for v in meta.values()]
    rated = [r for r in ratings if r is not None]
    coverage = len(rated) / len(ratings)
    assert 0.25 <= coverage <= 0.35
    dist = Counter(rated)
    # J 型：5 星占绝对多数，1 星多于中间分
    assert dist[5] > dist[4] + dist[3] + dist[2]
    assert dist[1] > dist[3]


def test_scenario_distribution_presale_dominant():
    """原始未配比状态：售前约 68%，占主导；售后/闲聊给足绝对数量供配比锚定。"""
    _, meta = generate_raw.generate_corpus(seed=42)
    dist = Counter(v["scenario"] for v in meta.values())
    presale = sum(v for k, v in dist.items() if k == "售前")
    assert presale / len(meta) >= 0.6
    # 售后、闲聊明显多于售前之外的一半以下但具备配比锚定量级
    assert dist["售后"] / len(meta) >= 0.15


def test_sensitive_info_detectable():
    """约 20% 会话含敏感信息，且可被正则检出（手机号/地址/订单号）。"""
    messages, meta = generate_raw.generate_corpus(seed=42)
    sensitive_sids = {sid for sid, v in meta.items() if v["has_sensitive"]}
    assert 0.12 <= len(sensitive_sids) / len(meta) <= 0.30
    hits = Counter()
    for m in messages:
        hits.update(scan_sensitive(m["text"]))
    assert hits["手机号"] > 0
    assert hits["地址"] > 0
    assert hits["订单号"] > 0


def test_unrated_ordered_signal_ratio():
    """约 25% 无星会话带 24h 成交标记。"""
    _, meta = generate_raw.generate_corpus(seed=42)
    unrated = [v for v in meta.values() if v["rating"] is None]
    ordered = sum(1 for v in unrated if v["ordered_within_24h"])
    assert 0.15 <= ordered / len(unrated) <= 0.35


def test_reproducible_with_seed():
    """固定种子可复现。"""
    m1, meta1 = generate_raw.generate_corpus(num_customers=20, seed=42)
    m2, meta2 = generate_raw.generate_corpus(num_customers=20, seed=42)
    assert m1 == m2
    assert meta1 == meta2
