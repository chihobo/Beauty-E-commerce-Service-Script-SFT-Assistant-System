# -*- coding: utf-8 -*-
"""在线链路测试（R12）：数字校验器 + 双条件闸门链路。

对应 test_cases.md 用例：T18 / T19 / T20。
- T18：价格咨询且接口有数且数字校验通过 -> 自动发送（另覆盖物流自动通道、售后模板自动发送、活动规则委托）；
- T19：编造数字拦截转草稿（另覆盖低置信走草稿、接口查不到软化拒答、无事实接口意图走草稿）；
- T20：过敏不适永远转人工，不进入生成。
意图分类 client 与生成器一律用桩依赖注入，商品/快递接口用真实模拟实现。
"""
import json
from datetime import date, datetime

from src.serving import fact_apis
from src.serving.pipeline import (
    HUMAN_HANDOFF_REPLY,
    OutputFactChecker,
    handle_query,
)
from src.serving.promo_knowledge import PromoKnowledgeStore
from tests.stub_client import StubClient


# ---------- 生成器桩：固定回复（忠实引用 facts 或编造数字由测试指定），记录每次调用 ----------
class StubGenerator:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls = []

    def generate(self, system, user, facts=None, **kwargs):
        self.calls.append({"system": system, "user": user, "facts": facts})
        return self._reply


def _intent_client(intent: str, confidence: float) -> StubClient:
    """意图分类桩：固定返回指定意图与置信度的 JSON。"""
    return StubClient(f'{{"intent": "{intent}", "confidence": {confidence}}}')


# ==================== 数字校验器 OutputFactChecker ====================

def test_extract_numbers_basic():
    """抽取阿拉伯数字：价格、满减组合（满99减10 抽出 99 与 10）、小数。"""
    assert OutputFactChecker.extract_numbers("现价 129 元，还能领满99减10的券") == ["129", "99", "10"]
    assert OutputFactChecker.extract_numbers("券后 129.5 元") == ["129.5"]


def test_extract_numbers_normalization():
    """归一化：千分位逗号、前导零、整数小数（129.0 -> 129）、全角数字。"""
    assert OutputFactChecker.extract_numbers("库存 1,200 件") == ["1200"]
    assert OutputFactChecker.extract_numbers("129.0 元") == ["129"]
    assert OutputFactChecker.extract_numbers("现价１２９元") == ["129"]


def test_collect_fact_numbers_from_dict():
    """合法数字集合来自接口 dict 的所有值：价格/原价/库存/券文案/商品名里的数字。"""
    facts = fact_apis.get_product_info("P001")  # 129/169/500/满99减10/玻尿酸保湿精华 30ml
    legal = OutputFactChecker.collect_fact_numbers(facts)
    assert {"129", "169", "500", "99", "10", "30"} <= legal


def test_verify_pass_and_unmatched():
    """回复数字 ⊆ 接口数字则通过；多一个数字即拦截并原样返回未匹配数字。"""
    facts = fact_apis.get_product_info("P003")  # 69/89/1200/无券
    passed, unmatched = OutputFactChecker.verify("宝子这款 69 元哦", facts)
    assert passed and unmatched == []
    passed, unmatched = OutputFactChecker.verify("宝子这款只要 59 元", facts)
    assert not passed and unmatched == ["59"]


# ==================== T18：自动发送 ====================

def test_price_query_auto_send(tmp_path):
    """T18：价格咨询高置信 + 接口有数 + 生成忠实引用 -> 自动发送，回复含 129。"""
    gen = StubGenerator("宝子这款现在活动价 129 元哦（原价 169），还能领满99减10的券，库存 500 件充足哒~")
    result = handle_query(
        "这个精华现在多少钱",
        product_id="P001",
        client=_intent_client("价格咨询", 0.95),
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "auto_send"
    assert "129" in result.reply
    assert result.intent_result.intent == "价格咨询"
    assert result.facts["price"] == 129
    assert result.fact_check == {"checked": True, "passed": True, "unmatched": []}
    # 生成时确实注入了接口事实
    assert gen.calls[0]["facts"]["price"] == 129


def test_logistics_query_auto_send(tmp_path):
    """T18：物流查询高置信 + 接口有数 -> 自动发送（物流数字 2~3 天在接口 eta 中）。"""
    gen = StubGenerator("宝子，你的包裹已揽收，快递已在上海青浦揽收，预计 2~3 天送达哦~")
    result = handle_query(
        "我的快递到哪了",
        order_id="O20260618002",
        intent_version="v1",
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "auto_send"
    assert result.facts["status"] == "已揽收"
    assert result.fact_check["passed"] is True


def test_after_sale_template_auto_send(tmp_path):
    """T18：售后政策为模板化答案意图，命中固定模板直接自动发送，不经过生成。"""
    gen = StubGenerator("不应被调用")
    result = handle_query(
        "怎么退货，运费谁出",
        intent_version="v1",
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "auto_send"
    assert "七天无理由" in result.reply
    assert gen.calls == []  # 模板化答案不调用生成器


# ==================== T19：拦截转草稿 ====================

def test_fabricated_number_blocked(tmp_path):
    """T19：生成编造接口外的"99" -> 数字校验失败，拦截转草稿并记录未匹配数字。

    用无券商品 P003：若用 P001，券文案"满99减10"里的 99 会使编造的"99"
    落入合法集合而漏拦截——这正是精确集合匹配的边界，测试须避开。
    """
    gen = StubGenerator("宝子这款现在只要 99 元，赶紧下单哦~")
    result = handle_query(
        "洁面乳多少钱",
        product_id="P003",
        client=_intent_client("价格咨询", 0.95),
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "draft"
    assert result.fact_check["checked"] is True
    assert result.fact_check["passed"] is False
    assert result.fact_check["unmatched"] == ["99"]
    assert "99" in result.reason  # 决策依据记录编造数字，演示界面展示


def test_low_confidence_goes_draft(monkeypatch, tmp_path):
    """T19：意图置信度低于阈值 -> 不满足双条件，走草稿通道且不调用事实接口。"""
    api_called = []
    original = fact_apis.get_product_info
    monkeypatch.setattr(
        fact_apis, "get_product_info",
        lambda pid: (api_called.append(pid), original(pid))[1],
    )
    gen = StubGenerator("宝子稍等，我帮您看一下哦~")
    result = handle_query(
        "现在买划算吗",
        product_id="P001",
        client=_intent_client("价格咨询", 0.5),
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "draft"
    assert api_called == []  # 低置信不调接口
    assert gen.calls[0]["facts"] is None  # 草稿通道无事实注入
    assert "阈值" in result.reason


def test_fact_not_found_soft_refusal(tmp_path):
    """T19：接口查不到 -> 降级草稿 + 软化拒答话术，不进入生成。"""
    gen = StubGenerator("不应被调用")
    result = handle_query(
        "这个多少钱",
        product_id="P999",  # 不存在的商品
        client=_intent_client("价格咨询", 0.95),
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "draft"
    assert result.reply == "宝子稍等，这款的价格我帮您确认一下哦~"
    assert result.facts is None
    assert gen.calls == []


def test_non_fact_intent_goes_draft(tmp_path):
    """T19：无事实接口的意图（产品咨询）即使高置信也走草稿通道。"""
    gen = StubGenerator("宝子，这款油皮也可以用的哦~")
    result = handle_query(
        "这款适合油皮吗",
        intent_version="v1",
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "draft"
    assert result.intent_result.intent == "产品咨询"
    assert gen.calls[0]["facts"] is None


# ==================== T20：高危永远转人工 ====================

def test_allergy_always_human(tmp_path):
    """T20：过敏表述 -> 永远转人工，安抚话术，生成器不被调用。"""
    gen = StubGenerator("不应被调用")
    result = handle_query(
        "用了两天过敏烂脸了",
        intent_version="v1",
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "human"
    assert result.intent_result.intent == "过敏不适"
    assert result.reply == HUMAN_HANDOFF_REPLY
    assert gen.calls == []  # 高危场景不进入生成


# ==================== 活动规则委托 ====================

PROMO_CONTENT = (
    "一、活动时间：10月31日20:00开始，全场跨店每满300减50。\n"
    "二、定金膨胀：预付定金10元抵20元，尾款11月10日20:00开始支付。\n"
)


def _build_promo_store(tmp_path) -> PromoKnowledgeStore:
    store = PromoKnowledgeStore(store_path=tmp_path / "promo_docs.json")
    store.upload_doc("双11规则", PROMO_CONTENT, "2026-10-20", "2026-11-11")
    return store


def test_promo_intent_delegates_to_store(tmp_path):
    """T18：活动规则意图委托大促规则模块，忠实引用条款 -> 自动发送。"""
    store = _build_promo_store(tmp_path)
    # 注意：回复不能写"双11"——条款原文没有该字样，"11"会被条款事实校验正确拦截
    gen = StubGenerator("宝子，活动10月31日20:00开始，全场跨店每满300减50哦~")
    result = handle_query(
        "双11满减能叠加吗",
        intent_version="v1",
        generator=gen,
        promo_store=store,
        now=date(2026, 11, 1),
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "auto_send"
    assert result.fact_check["passed"] is True
    assert result.facts["命中条款"]  # 命中条款已注入生成上下文


def test_promo_intent_without_store_goes_draft(tmp_path):
    """T19：活动规则意图但未配置规则存储 -> 无事实来源，走草稿通道。"""
    gen = StubGenerator("宝子，活动规则我帮您整理一下哦~")
    result = handle_query(
        "双11满减能叠加吗",
        intent_version="v1",
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "draft"
    assert gen.calls[0]["facts"] is None


def test_promo_expired_goes_human(tmp_path):
    """活动规则过期 -> 大促模块检索为空转人工，安抚话术，生成器不被调用。"""
    store = _build_promo_store(tmp_path)
    gen = StubGenerator("不应被调用")
    result = handle_query(
        "双11满减能叠加吗",
        intent_version="v1",
        generator=gen,
        promo_store=store,
        now=date(2026, 11, 12),
        log_path=tmp_path / "log.jsonl",
    )
    assert result.decision == "human"
    assert result.reply == HUMAN_HANDOFF_REPLY
    assert gen.calls == []


# ==================== 决策日志 ====================

def test_decision_log_appended(tmp_path):
    """决策日志追加写 jsonl：时间戳/query/intent/decision/reason 字段齐全。"""
    log_file = tmp_path / "decision_log.jsonl"
    gen = StubGenerator("宝子这款现在活动价 129 元哦~")
    handle_query(
        "这个精华多少钱",
        product_id="P001",
        client=_intent_client("价格咨询", 0.95),
        generator=gen,
        now=datetime(2026, 10, 22, 12, 30, 0),
        log_path=log_file,
    )
    handle_query(
        "用了两天过敏烂脸了",
        intent_version="v1",
        generator=gen,
        now=datetime(2026, 10, 22, 12, 31, 0),
        log_path=log_file,
    )
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2  # 追加写而非覆盖
    first, second = (json.loads(line) for line in lines)
    assert first["timestamp"] == "2026-10-22T12:30:00"
    assert first["query"] == "这个精华多少钱"
    assert first["intent"] == "价格咨询"
    assert first["decision"] == "auto_send"
    assert first["reason"]
    assert second["intent"] == "过敏不适"
    assert second["decision"] == "human"


def test_generator_receives_brand_system_prompt(tmp_path):
    """生成请求携带品牌人设 system prompt（称呼宝子、事实靠注入等硬约束）。"""
    gen = StubGenerator("宝子这款现在 129 元哦~")
    handle_query(
        "多少钱",
        product_id="P001",
        client=_intent_client("价格咨询", 0.95),
        generator=gen,
        log_path=tmp_path / "log.jsonl",
    )
    system = gen.calls[0]["system"]
    assert "宝子" in system
    assert "禁止编造" in system
