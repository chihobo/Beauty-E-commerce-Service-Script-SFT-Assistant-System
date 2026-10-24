# -*- coding: utf-8 -*-
"""在线链路串联（R12）：意图分类 -> 高危拦截 -> 双条件闸门 -> 事实注入生成 -> 数字校验 -> 自动发送/草稿/人工。

对应面试叙事：
- 双条件闸门：自动发送必须同时满足「高置信意图 + 该意图有事实接口」，
  其余全部留在草稿模式——系统对不确定的事默认交还给人；
- 数字校验：回复里的数字必须都能在接口返回值里找到，多一个数字就拦截，
  从机制上杜绝报错价（微调管行为，接口管知识）；
- 高危场景（过敏不适）永远转人工，不进入生成；
- 拒答不生硬说"不知道"，品牌语气安抚 + 转人工。

与 promo_knowledge.FactChecker 的区别：那里的事实源是规则条款文本，
这里 OutputFactChecker 的事实源是接口返回的 dict——把 dict 所有值里的
数字收集为合法集合，再做精确集合比对。
"""
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

from src.config import DATA_PROCESSED, INTENT_CONFIDENCE_THRESHOLD
from src.serving import fact_apis
from src.serving.intent import HIGH_RISK_INTENTS, IntentResult, classify

# 品牌人设 system prompt（与训练语料的 system 设定一致：称呼宝子、emoji 限量、事实靠注入）
BRAND_SYSTEM_PROMPT = (
    "你是某国产美妆品牌旗舰店的金牌客服。"
    "硬性要求：称呼客户为「宝子」，语气亲昵活泼；每条回复 emoji 不超过 3 个；"
    "价格、库存、物流等事实只能使用注入的接口数据，禁止编造接口之外的数字；"
    "不得使用美白、祛斑、治疗等违规宣称用语。"
)

# 安抚转人工话术：高危拦截与各类转人工场景统一口径（不生硬说"不知道"）
HUMAN_HANDOFF_REPLY = "宝子稍等，这个我帮您转接专业顾问确认一下哦~"

# 有事实接口的意图 -> 接口查不到时的软化拒答话术（转草稿，人工跟进）
FACT_API_SOFT_REFUSAL = {
    "价格咨询": "宝子稍等，这款的价格我帮您确认一下哦~",
    "物流查询": "宝子稍等，这个订单的物流信息我帮您核实一下哦~",
}

# 售后政策模板表：模板化答案意图，固定内容直接自动发送，不经过生成
# 每条为 (命中关键词正则, 模板名, 模板话术)；模板统一用中文数字，与数字校验的"只校验阿拉伯数字"约定一致
AFTER_SALE_TEMPLATES = [
    (
        re.compile(r"退|换货|七天无理由"),
        "退换货",
        "宝子，咱家支持七天无理由退换货哦~ 签收后七天内、不影响二次销售就能申请，"
        "在订单页点「申请售后」就行啦，退款会在售后完成后原路返回哒~",
    ),
    (
        re.compile(r"运费|邮费"),
        "运费",
        "宝子，运费是这样哒：商品质量问题退换，来回运费咱家承担；"
        "七天无理由退换的话，运费需要宝子自理哦~",
    ),
    (
        re.compile(r"保修|质保|维修"),
        "保修",
        "宝子放心，咱家商品都有质保服务哒，有质量问题随时联系我们，一定帮您妥善处理哦~",
    ),
]
# 未命中具体关键词时的售后通用模板
AFTER_SALE_DEFAULT = (
    "通用",
    "宝子，售后问题都可以找我哦~ 退换货、运费、保修都能帮您处理哒，跟我讲讲遇到什么情况啦~",
)


def _to_half_width(text: str) -> str:
    """全角字符转半角。

    刻意保留空白字符（不像 promo_knowledge.FactChecker._normalize 那样去空白）：
    数字校验按 token 抽取，若去掉空白会把空格隔开的两个独立数字粘成一个。
    """
    chars = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角符号（含全角数字）转半角
            chars.append(chr(code - 0xFEE0))
        else:
            chars.append(ch)
    return "".join(chars)


class OutputFactChecker:
    """输出数字校验器：回复中的每个阿拉伯数字都必须能在接口返回 facts 中找到。

    校验口径与 promo_knowledge.FactChecker 一致：精确集合匹配，不做子串匹配
    （否则编造的"减 30"会被合法集合里的"满 300"误吞而漏拦截）；
    中文数字（"两件""三天"）刻意不校验，语义等价判断留给人工草稿环节。
    """

    # 阿拉伯数字（含小数、千分位逗号）；"满99减10"这类满减组合按 99、10 两个 token 抽出
    _NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")

    @staticmethod
    def _canonical(token: str) -> str:
        """数字 token 归一化：去千分位逗号、去前导零、"129.0" 归一为 "129"。"""
        token = token.replace(",", "")
        if "." in token:
            head, tail = token.split(".", 1)
            tail = tail.rstrip("0")
            token = head if not tail else f"{head}.{tail}"
        return token.lstrip("0") or "0"

    @classmethod
    def extract_numbers(cls, text: str) -> list[str]:
        """抽取文本中的阿拉伯数字，半角归一化后去重保序返回。"""
        normalized = _to_half_width(text)
        numbers = []
        for m in cls._NUMBER_PATTERN.finditer(normalized):
            number = cls._canonical(m.group(0))
            if number not in numbers:
                numbers.append(number)
        return numbers

    @classmethod
    def collect_fact_numbers(cls, facts: dict) -> set[str]:
        """把接口返回 dict 所有值里的数字收集为合法集合。

        数值型直接取值；字符串（商品名如"...30ml"、券文案如"满99减10"）
        与字符串列表（券列表）逐一抽取其中的数字。
        """
        legal = set()
        for value in facts.values():
            if isinstance(value, bool):  # bool 是 int 子类，排除避免 True/False 混进数字集
                continue
            if isinstance(value, (int, float)):
                legal.add(cls._canonical(str(value)))
            elif isinstance(value, str):
                legal.update(cls.extract_numbers(value))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    legal.update(cls.extract_numbers(str(item)))
        return legal

    @classmethod
    def verify(cls, reply: str, facts: dict) -> tuple[bool, list[str]]:
        """校验回复中的每个数字都属于接口事实集合，返回 (是否通过, 未匹配数字列表)。"""
        legal = cls.collect_fact_numbers(facts)
        unmatched = [n for n in cls.extract_numbers(reply) if n not in legal]
        return (not unmatched, unmatched)


@dataclass
class PipelineResult:
    """在线链路决策结果。

    decision: "auto_send"（自动发送）/ "draft"（转草稿人工处理）/ "human"（转人工）；
    fact_check 记录数字校验明细：checked（是否执行了校验）、passed（校验结果）、
    unmatched（未匹配数字），演示界面按此展示校验环节；
    reason 串联各环节决策依据，演示界面要展示完整链路。
    """
    decision: str
    reply: str | None
    intent_result: IntentResult
    facts: dict | None
    fact_check: dict
    reason: str


def _match_after_sale_template(query: str) -> tuple[str, str]:
    """按关键词命中售后政策模板，返回 (模板名, 模板话术)；未命中返回通用模板。"""
    for pattern, name, template in AFTER_SALE_TEMPLATES:
        if pattern.search(query):
            return name, template
    return AFTER_SALE_DEFAULT


def _append_decision_log(log_path: Path, query: str, result: PipelineResult, now=None) -> None:
    """追加写决策日志（jsonl），Streamlit 看板按此统计采纳率与拦截记录。"""
    moment = now if now else datetime.now()
    if isinstance(moment, datetime):
        timestamp = moment.isoformat(timespec="seconds")
    elif isinstance(moment, date):
        timestamp = datetime.combine(moment, time.min).isoformat(timespec="seconds")
    else:
        timestamp = str(moment)
    record = {
        "timestamp": timestamp,
        "query": query,
        "intent": result.intent_result.intent,
        "confidence": result.intent_result.confidence,
        "decision": result.decision,
        "reason": result.reason,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _no_check() -> dict:
    """未执行数字校验时的 fact_check 占位（无生成或无事实注入的环节）。"""
    return {"checked": False, "passed": None, "unmatched": []}


def handle_query(
    query: str,
    product_id: str | None = None,
    order_id: str | None = None,
    intent_version: str = "v2",
    client=None,
    generator=None,
    promo_store=None,
    now=None,
    log_path: Path | None = None,
) -> PipelineResult:
    """在线链路主入口。

    依赖注入：client（意图分类 LLM）、generator（推理服务客户端，须有
    generate(system, user, facts) 方法）、promo_store（大促规则存储），
    测试一律用桩；log_path 默认写 data/processed/decision_log.jsonl。
    """
    resolved_log_path = Path(log_path) if log_path else DATA_PROCESSED / "decision_log.jsonl"

    def _finish(result: PipelineResult) -> PipelineResult:
        _append_decision_log(resolved_log_path, query, result, now=now)
        return result

    # 环节 1：意图分类
    intent_result = classify(query, version=intent_version, client=client)
    intent = intent_result.intent
    confidence = intent_result.confidence
    chain_reason = (
        f"意图「{intent}」（{intent_version} 版，置信度 {confidence:.2f}，{intent_result.reason}）"
    )

    # 环节 2：高危拦截——过敏不适永远转人工，不进入生成
    if intent in HIGH_RISK_INTENTS:
        return _finish(PipelineResult(
            decision="human",
            reply=HUMAN_HANDOFF_REPLY,
            intent_result=intent_result,
            facts=None,
            fact_check=_no_check(),
            reason=f"{chain_reason}；命中高危意图，永远转人工，不进入生成",
        ))

    # 环节 3：售后政策——模板化答案意图，固定模板自动发送，不经过生成与数字校验
    if intent == "售后政策":
        template_name, reply = _match_after_sale_template(query)
        return _finish(PipelineResult(
            decision="auto_send",
            reply=reply,
            intent_result=intent_result,
            facts=None,
            fact_check=_no_check(),
            reason=f"{chain_reason}；售后政策为模板化答案意图，命中「{template_name}」模板，自动发送",
        ))

    # 环节 4：活动规则——委托大促规则模块（检索 -> 注入生成 -> 条款事实校验）；
    # 未传入 promo_store 时该意图无事实来源，走草稿通道
    if intent == "活动规则" and promo_store is not None:
        promo_result = promo_store.handle_promo_query(
            query,
            lambda q, clauses: generator.generate(
                system=BRAND_SYSTEM_PROMPT, user=q, facts={"活动规则": "\n".join(clauses)}
            ),
            now=now,
        )
        if promo_result["decision"] == "human":
            fact_check = _no_check()
            reply = HUMAN_HANDOFF_REPLY
        else:
            fact_check = {
                "checked": True,
                "passed": promo_result["decision"] == "auto_send",
                "unmatched": promo_result["unmatched_facts"],
            }
            reply = promo_result["reply"]
        return _finish(PipelineResult(
            decision=promo_result["decision"],
            reply=reply,
            intent_result=intent_result,
            facts={"命中条款": promo_result["clauses"]},
            fact_check=fact_check,
            reason=f"{chain_reason}；委托大促规则模块：{promo_result['reason']}",
        ))

    # 环节 5：双条件闸门——有事实接口的意图（价格咨询/物流查询）且置信度达标，进入自动通道
    if intent in FACT_API_SOFT_REFUSAL and confidence >= INTENT_CONFIDENCE_THRESHOLD:
        if intent == "价格咨询":
            facts = fact_apis.get_product_info(product_id) if product_id else None
        else:
            facts = fact_apis.get_logistics(order_id) if order_id else None
        gate_reason = (
            f"{chain_reason}；双条件通过（该意图有事实接口，"
            f"置信度 {confidence:.2f} ≥ 阈值 {INTENT_CONFIDENCE_THRESHOLD}）"
        )

        # 接口查不到 -> 降级草稿，软化拒答话术（不生成，固定话术）
        if facts is None:
            return _finish(PipelineResult(
                decision="draft",
                reply=FACT_API_SOFT_REFUSAL[intent],
                intent_result=intent_result,
                facts=None,
                fact_check=_no_check(),
                reason=f"{gate_reason}；接口查不到数据，降级草稿，软化拒答转人工确认",
            ))

        # 事实注入生成
        reply = generator.generate(system=BRAND_SYSTEM_PROMPT, user=query, facts=facts)

        # 环节 6：输出数字校验——回复数字必须都能在接口返回值中找到
        passed, unmatched = OutputFactChecker.verify(reply, facts)
        fact_check = {"checked": True, "passed": passed, "unmatched": unmatched}
        if passed:
            return _finish(PipelineResult(
                decision="auto_send",
                reply=reply,
                intent_result=intent_result,
                facts=facts,
                fact_check=fact_check,
                reason=f"{gate_reason}；接口取数成功，生成回复数字校验通过，自动发送",
            ))
        return _finish(PipelineResult(
            decision="draft",
            reply=reply,
            intent_result=intent_result,
            facts=facts,
            fact_check=fact_check,
            reason=f"{gate_reason}；回复含接口外的编造数字 {unmatched}，拦截转草稿",
        ))

    # 环节 7：草稿通道——置信度不足，或该意图没有事实接口（产品咨询/闲聊安抚/其他/
    # 未配置规则存储的活动规则），生成草稿（无事实注入）由人工把关
    reply = generator.generate(system=BRAND_SYSTEM_PROMPT, user=query, facts=None)
    if intent in FACT_API_SOFT_REFUSAL:
        draft_reason = (
            f"置信度 {confidence:.2f} 低于阈值 {INTENT_CONFIDENCE_THRESHOLD}，"
            "不满足双条件，走草稿通道"
        )
    else:
        draft_reason = "该意图无事实接口（或活动规则未配置规则存储），不满足双条件，走草稿通道"
    return _finish(PipelineResult(
        decision="draft",
        reply=reply,
        intent_result=intent_result,
        facts=None,
        fact_check=_no_check(),
        reason=f"{chain_reason}；{draft_reason}",
    ))
