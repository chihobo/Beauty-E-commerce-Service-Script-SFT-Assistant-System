# -*- coding: utf-8 -*-
"""意图分类模块：V1 关键词规则版 + V2 LLM 前置分类版。

对应面试叙事：
- 第一版关键词规则上线后发现客户会绕开关键词（"现在买划算吗"没有"价格"俩字），
  覆盖率只有七八成；
- 第二版改 LLM 前置分类，一次调用几分钱，准确率 95%+。
两版都保留，可切换对比，复现"踩坑 -> 迭代"的故事。

8 类意图（与简历问答一致）：
价格咨询 / 物流查询 / 活动规则 / 产品咨询 / 售后政策 / 过敏不适 / 闲聊安抚 / 其他。
"""
import json
import re
from dataclasses import dataclass

from ..config import LLM_MODEL

# 8 类意图完整集合，V2 的 prompt 与结果校验都以此为准
INTENTS = [
    "价格咨询",
    "物流查询",
    "活动规则",
    "产品咨询",
    "售后政策",
    "过敏不适",
    "闲聊安抚",
    "其他",
]

# 高危意图：命中后永远转人工，不进入自动生成（双条件闸门在链路层处理）
HIGH_RISK_INTENTS = {"过敏不适"}

# V2 分类时给 LLM 的意图定义说明（prompt 中逐条列出）
INTENT_DEFINITIONS = {
    "价格咨询": "问价格、优惠、券、划算不划算、怎么买更便宜",
    "物流查询": "问快递到哪、什么时候发货、什么时候能到",
    "活动规则": "大促满减叠加、定金尾款、跨店优惠等活动规则解释",
    "产品咨询": "肤质匹配、成分、色号、用法、功效等产品问题",
    "售后政策": "退换货流程、保修、运费、退款等售后问题",
    "过敏不适": "过敏、烂脸、刺激、红肿、起疹子等使用不适反馈",
    "闲聊安抚": "客户发泄情绪、抱怨、闲聊，需要安抚",
    "其他": "无法归入以上任何一类，或拿不准",
}


@dataclass
class IntentResult:
    """意图分类结果。

    confidence 取值 0~1；version 标记来源版本，演示界面用于对比展示；
    reason 记录判定依据（命中的关键词 / LLM 原始返回），演示界面要展示。
    """
    intent: str
    confidence: float
    version: str
    reason: str


# V1 关键词规则表：高危词（过敏不适）放最前优先匹配，避免被其他类抢命中
# 每项为 (意图, 关键词正则, 规则说明)
_V1_RULES = [
    ("过敏不适", r"过敏|烂脸|刺激|红肿|起疹|刺痛", "命中高危词"),
    ("价格咨询", r"多少钱|价格|优惠|券|便宜|降价|打折", "命中价格关键词"),
    ("物流查询", r"物流|快递|发货|到哪|几天到|什么时候发", "命中物流关键词"),
    ("活动规则", r"满减|叠加|定金|尾款|跨店|双\s*11|双十一|618|大促", "命中活动规则关键词"),
    ("产品咨询", r"肤质|成分|色号|怎么用|用法|功效|油皮|干皮|敏感皮", "命中产品关键词"),
    ("售后政策", r"退|换货|退款|运费|保修|七天无理由", "命中售后关键词"),
    ("闲聊安抚", r"生气|投诉|差评|无语|气死|烦", "命中情绪安抚关键词"),
]


def classify_v1(text: str) -> IntentResult:
    """V1 关键词规则版。

    逐条规则按优先级匹配，命中即 confidence=1.0（规则版没有概率概念）；
    未命中任何规则判为"其他"、confidence=0，交给上层走转人工/草稿分支。

    已知局限（面试踩坑点）：客户绕开关键词的问法（如"现在买划算吗"）
    不含任何关键词，会被漏判为"其他"，这正是迭代出 V2 的原因。
    """
    for intent, pattern, rule_desc in _V1_RULES:
        match = re.search(pattern, text)
        if match:
            return IntentResult(
                intent=intent,
                confidence=1.0,
                version="v1_rules",
                reason=f"{rule_desc}「{match.group(0)}」",
            )
    return IntentResult(
        intent="其他",
        confidence=0.0,
        version="v1_rules",
        reason="未命中任何关键词规则",
    )


def classify_v2(text: str, client) -> IntentResult:
    """V2 LLM 前置分类版。

    client 为 OpenAI 兼容客户端，依赖注入（测试用桩）。
    要求模型返回 JSON {"intent": "...", "confidence": 0.xx}；
    解析失败或意图不在 8 类集合内时按"其他"处理，reason 记录失败原因。

    项目约定不做兜底链路：这里"按其他处理"不是兜底，而是业务上
    "不确定即交还给人"的既定策略（与置信度低走草稿/人工一致）。
    """
    intent_lines = "\n".join(
        f"- {name}：{desc}" for name, desc in INTENT_DEFINITIONS.items()
    )
    prompt = (
        "你是美妆电商客服的意图分类器。请把客户消息分类到以下 8 类意图之一：\n"
        f"{intent_lines}\n\n"
        '只返回 JSON，格式：{"intent": "意图名", "confidence": 0到1之间的置信度}，'
        "不要输出任何其他内容。\n\n"
        f"客户消息：{text}"
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        # 容忍模型在 JSON 外面包 markdown 代码块的情况
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(json_match.group(0) if json_match else raw)
        intent = parsed["intent"]
        confidence = float(parsed["confidence"])
        if intent not in INTENTS:
            raise ValueError(f"意图「{intent}」不在 8 类集合内")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"置信度 {confidence} 超出 0~1 范围")
        return IntentResult(
            intent=intent,
            confidence=confidence,
            version="v2_llm",
            reason=f"LLM 判定（原始返回：{raw.strip()}）",
        )
    except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as exc:
        return IntentResult(
            intent="其他",
            confidence=0.0,
            version="v2_llm",
            reason=f"LLM 返回解析失败（{exc}），原始返回：{raw.strip()!r}，按「其他」处理",
        )


def classify(text: str, version: str = "v2", client=None) -> IntentResult:
    """统一入口：按 version 选择分类版本。

    V2 必须显式传入 client（依赖注入，不在此处读取环境变量），
    未传入即视为调用方配置错误，直接报错，不做静默降级到 V1。
    """
    if version == "v1":
        return classify_v1(text)
    if version == "v2":
        if client is None:
            raise ValueError("V2 意图分类必须注入 LLM client")
        return classify_v2(text, client)
    raise ValueError(f"未知的意图分类版本：{version}（仅支持 v1 / v2）")
