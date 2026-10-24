# -*- coding: utf-8 -*-
"""T4 脱敏测试（对应 T02、T03 / R2）。"""
from collections import Counter

from src.data_pipeline.desensitize import (
    DESENSITIZE_RULES,
    desensitize_text,
    scan_sensitive,
)


def test_rules_centralized_and_configurable():
    """脱敏规则集中在 DESENSITIZE_RULES，含名称/正则/占位符三要素。"""
    names = [name for name, _, _ in DESENSITIZE_RULES]
    assert "手机号" in names
    assert "订单号" in names
    assert "地址" in names
    for name, pattern, placeholder in DESENSITIZE_RULES:
        assert name and placeholder.startswith("[") and placeholder.endswith("]")
        assert hasattr(pattern, "subn")


def test_phone_replaced():
    counter = Counter()
    out = desensitize_text("快递到了打我手机13812345678哦", counter=counter)
    assert "13812345678" not in out
    assert "[手机号]" in out
    assert counter["手机号"] == 1


def test_order_id_replaced():
    counter = Counter()
    out = desensitize_text("我的订单号是202410158888666，帮我查下", counter=counter)
    assert "202410158888666" not in out
    assert "[订单号]" in out
    assert counter["订单号"] == 1


def test_address_replaced():
    counter = Counter()
    out = desensitize_text("收货地址：浙江省杭州市西湖区文三路128号3幢502室，别发错", counter=counter)
    assert "文三路" not in out
    assert "[地址]" in out
    assert counter["地址"] == 1


def test_nickname_replaced():
    counter = Counter()
    out = desensitize_text("收件人写桃桃酱就行", nickname="桃桃酱", counter=counter)
    assert "桃桃酱" not in out
    assert "[买家昵称]" in out
    assert counter["买家昵称"] == 1


def test_scan_clean_after_desensitize():
    """脱敏后全量扫描无任何敏感信息命中（T03）。"""
    raw = (
        "电话13812345678，地址：江苏省南京市鼓楼区中山北路66号，"
        "订单202503027777888，收件人桃桃酱"
    )
    assert set(scan_sensitive(raw, nicknames={"桃桃酱"})) == {"手机号", "地址", "订单号", "买家昵称"}
    out = desensitize_text(raw, nickname="桃桃酱")
    assert scan_sensitive(out, nicknames={"桃桃酱"}) == []
