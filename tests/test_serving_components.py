# -*- coding: utf-8 -*-
"""在线服务组件测试：意图分类两版（R10）、模拟事实接口（R11）、大促规则模块（R13）。

对应 test_cases.md 用例：T14 / T15 / T16 / T17 / T21 / T22 / T23。
LLM 与生成器一律用桩依赖注入，不发真实请求。
"""
from datetime import date
from types import SimpleNamespace

from src.serving import fact_apis
from src.serving.intent import classify, classify_v1, classify_v2
from src.serving.promo_knowledge import (
    FULL_INJECTION_CHAR_LIMIT,
    FactChecker,
    PromoKnowledgeStore,
)


# ---------- LLM 桩：返回预设 content，模拟 OpenAI 兼容客户端 ----------
def _make_stub_client(content: str):
    def _create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))


# ==================== R10 意图分类两版 ====================

def test_v1_rules_hit_each_intent():
    """T14：V1 关键词规则对各意图的命中。"""
    cases = [
        ("这个精华多少钱", "价格咨询"),
        ("我的快递到哪了", "物流查询"),
        ("双11满减能叠加吗", "活动规则"),
        ("这款适合油皮吗", "产品咨询"),
        ("怎么退货，运费谁出", "售后政策"),
        ("用了两天过敏烂脸了", "过敏不适"),
        ("气死了，我要给差评", "闲聊安抚"),
    ]
    for text, expected in cases:
        result = classify_v1(text)
        assert result.intent == expected, f"输入「{text}」期望「{expected}」，实际「{result.intent}」"
        assert result.confidence == 1.0
        assert result.version == "v1_rules"
        assert result.reason  # 判定依据必须非空，演示界面要展示


def test_v1_misses_keyword_evasion():
    """T15 前半：客户绕开关键词（"现在买划算吗"无价格关键词），V1 漏判为"其他"。"""
    result = classify_v1("现在买划算吗")
    assert result.intent == "其他"
    assert result.confidence == 0.0


def test_v2_llm_corrects_keyword_evasion():
    """T15 后半：V2（桩）把"现在买划算吗"正确判为"价格咨询"并带置信度。"""
    client = _make_stub_client('{"intent": "价格咨询", "confidence": 0.92}')
    result = classify_v2("现在买划算吗", client)
    assert result.intent == "价格咨询"
    assert result.confidence == 0.92
    assert result.version == "v2_llm"


def test_v2_parse_failure_falls_to_other():
    """V2 桩返回非 JSON 垃圾内容时，按"其他"处理且 reason 记录解析失败。"""
    client = _make_stub_client("我觉得这个问题嘛……不太好说")
    result = classify_v2("现在买划算吗", client)
    assert result.intent == "其他"
    assert result.confidence == 0.0
    assert "解析失败" in result.reason


def test_v2_invalid_intent_name_falls_to_other():
    """V2 桩返回 8 类集合外的意图名时，按"其他"处理。"""
    client = _make_stub_client('{"intent": "砍价", "confidence": 0.9}')
    result = classify_v2("能不能便宜点", client)
    assert result.intent == "其他"


def test_classify_unified_entry():
    """统一入口：v1 走规则、v2 必须注入 client。"""
    assert classify("多少钱", version="v1").intent == "价格咨询"
    client = _make_stub_client('{"intent": "价格咨询", "confidence": 0.95}')
    assert classify("现在买划算吗", version="v2", client=client).intent == "价格咨询"


# ==================== R11 模拟事实接口 ====================

def test_get_product_info_found():
    """T16：商品 API 返回现价/原价/库存/生效券预设结构。"""
    info = fact_apis.get_product_info("P001")
    assert info is not None
    assert info["price"] == 129
    assert info["original_price"] == 169
    assert info["stock"] == 500
    assert info["coupons"] == ["满99减10"]


def test_get_product_info_not_found():
    """T16：查不到的商品 ID 返回 None，供链路走拒答分支。"""
    assert fact_apis.get_product_info("P999") is None


def test_get_logistics_found_and_not_found():
    """T17：快递 API 返回物流状态/最新节点/预计送达；查不到返回 None。"""
    info = fact_apis.get_logistics("O20260618001")
    assert info is not None
    assert info["status"] == "运输中"
    assert info["latest"] == "已到杭州转运中心"
    assert info["eta"] == "明天送达"
    assert fact_apis.get_logistics("O00000000000") is None


# ==================== R13 大促规则模块 ====================

# 双 11 规则样例：覆盖中文序号与阿拉伯序号两种条款编号
PROMO_CONTENT = (
    "一、活动时间：10月31日20:00开始，全场跨店每满300减50。\n"
    "二、定金膨胀：预付定金10元抵20元，尾款11月10日20:00开始支付。\n"
    "3. 会员专享：店铺会员额外领满99减10券，限11月11日当天使用。\n"
)
PROMO_START = "2026-10-20"
PROMO_END = "2026-11-11"


def _build_store(tmp_path) -> PromoKnowledgeStore:
    store = PromoKnowledgeStore(store_path=tmp_path / "promo_docs.json")
    store.upload_doc("双11规则", PROMO_CONTENT, PROMO_START, PROMO_END)
    return store


def test_upload_doc_splits_clauses(tmp_path):
    """T21 前半：上传规则文档按条款编号切块，并持久化为 JSON。"""
    store = _build_store(tmp_path)
    clauses = store.upload_doc("双11规则", PROMO_CONTENT, PROMO_START, PROMO_END)
    assert len(clauses) == 3
    assert clauses[0].startswith("一、")
    assert clauses[1].startswith("二、")
    assert clauses[2].startswith("3.")
    assert "满300减50" in clauses[0]
    # JSON 持久化：换新实例重载，条款仍在
    reloaded = PromoKnowledgeStore(store_path=tmp_path / "promo_docs.json")
    assert reloaded.retrieve("满减", now=date(2026, 11, 1)) == clauses


def test_retrieve_within_validity(tmp_path):
    """T21 后半：有效期内检索命中条款（总字数小于阈值走全量注入）。"""
    store = _build_store(tmp_path)
    clauses = store.retrieve("满减怎么叠加", now=date(2026, 11, 1))
    assert len(clauses) == 3  # 一页纸规则总字数 < 1500，全量注入


def test_retrieve_after_expiry_returns_empty(tmp_path):
    """T23 前半：有效期过后条款自动下线，检索为空。"""
    store = _build_store(tmp_path)
    assert store.retrieve("满减怎么叠加", now=date(2026, 11, 12)) == []


def test_retrieve_before_start_returns_empty(tmp_path):
    """有效期开始前同样不可见。"""
    store = _build_store(tmp_path)
    assert store.retrieve("满减", now=date(2026, 10, 19)) == []


def test_retrieve_similarity_path_when_over_limit(tmp_path):
    """条款总字数超过阈值时退到 2-gram 相似度，取 top3 且最相关的排第一。"""
    filler_a = "甲" * 600
    filler_b = "乙" * 600
    filler_c = "丙" * 600
    filler_d = "丁" * 600
    content = (
        f"一、{filler_a}\n"
        f"二、{filler_b}\n"
        f"三、{filler_c}\n"
        f"四、{filler_d}\n"
    )
    assert len(content) > FULL_INJECTION_CHAR_LIMIT
    store = PromoKnowledgeStore(store_path=tmp_path / "promo_docs.json")
    store.upload_doc("长规则", content, "2026-10-20", "2026-11-11")
    clauses = store.retrieve("甲甲", now=date(2026, 11, 1))
    assert len(clauses) == 3  # top3
    assert clauses[0].startswith("一、")


def test_cleanup_expired_removes_doc(tmp_path):
    """T23：cleanup_expired 物理清理过期文档，清理后检索为空。"""
    store = _build_store(tmp_path)
    removed = store.cleanup_expired(now=date(2026, 11, 12))
    assert removed == ["双11规则"]
    assert store.retrieve("满减", now=date(2026, 11, 1)) == []  # 已被物理删除


def test_fact_checker_faithful_reply_passes():
    """T22 对照组：忠实引用条款的回复通过校验。"""
    clauses = ["一、活动时间：10月31日20:00开始，全场跨店每满300减50。"]
    reply = "活动在10月31日20:00开始，全场每满300减50哦"
    passed, unmatched = FactChecker.verify(reply, clauses)
    assert passed
    assert unmatched == []


def test_fact_checker_fabricated_numbers_blocked():
    """T22：编造条款外的"满200减30"被拦截，未匹配事实原样返回。"""
    clauses = ["一、活动时间：10月31日20:00开始，全场跨店每满300减50。"]
    reply = "现在下单满200减30，超级划算"
    passed, unmatched = FactChecker.verify(reply, clauses)
    assert not passed
    assert "200" in unmatched
    assert "30" in unmatched


def test_fact_checker_normalization():
    """全角转半角、去空格："20：00"与"20:00"等价，"10月31日"按整体日期事实抽取。"""
    assert FactChecker._normalize("２０：００") == "20:00"
    facts = FactChecker.extract_facts("10月31日 20：00 开抢")
    assert "10月31日" in facts
    assert "20:00" in facts
    # 全角写法回复对半角条款仍通过
    clauses = ["活动时间：10月31日20:00开始"]
    passed, _ = FactChecker.verify("活动 20：00 开始", clauses)
    assert passed


def test_handle_promo_query_auto_send(tmp_path):
    """T21/T22 链路：忠实引用条款的生成桩，校验通过自动发送。"""
    store = _build_store(tmp_path)
    faithful = lambda query, clauses: f"按活动规则，{clauses[0]}"
    result = store.handle_promo_query("双11满减怎么算", faithful, now=date(2026, 11, 1))
    assert result["decision"] == "auto_send"
    assert result["reply"].startswith("按活动规则")
    assert result["clauses"]  # 条款已注入生成上下文
    assert result["unmatched_facts"] == []


def test_handle_promo_query_draft_on_fabrication(tmp_path):
    """T22 链路：编造"满200减30"的生成桩被数字校验拦截转草稿。"""
    store = _build_store(tmp_path)
    fabricating = lambda query, clauses: "现在下单满200减30，赶紧抢"
    result = store.handle_promo_query("双11满减怎么算", fabricating, now=date(2026, 11, 1))
    assert result["decision"] == "draft"
    assert "200" in result["unmatched_facts"]
    assert "30" in result["unmatched_facts"]


def test_handle_promo_query_human_when_expired(tmp_path):
    """T23 链路：有效期过后检索为空，直接转人工，不进入生成。"""
    store = _build_store(tmp_path)
    called = []

    def generator(query, clauses):
        called.append(True)
        return "不应被调用"

    result = store.handle_promo_query("双11满减怎么算", generator, now=date(2026, 11, 12))
    assert result["decision"] == "human"
    assert result["reply"] is None
    assert called == []  # 无有效条款时生成器不被调用
