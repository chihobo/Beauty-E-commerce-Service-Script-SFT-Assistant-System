# -*- coding: utf-8 -*-
"""
千牛原始会话 -> SFT 训练数据清洗管线

对应《美妆电商客服话术 SFT 微调》数据工程流程:
  1. 消息流水按 30 分钟间隔拆会话(session)
  2. 脱敏: 手机号/地址/订单号/买家昵称 -> 占位符(入库前完成, 合规前置)
  3. 关联满意度评分与 24h 成交信号
  4. 质量筛选: 基础门槛 + 打星快速通道 + 成交信号 + LLM 判别(此处用关键词规则演示判别位置)
  5. 场景分类: 售前/售后/安抚(生产用 LLM 批量打标, 此处用关键词规则演示)
  6. 事实与风格分离: 价格/券/物流/保质期等易变事实 -> 占位符, 不进模型记忆
  7. 60/30/10 配比调整(售前降采样, 售后与安抚全保留), 导出 SFT messages JSONL

用法: python scripts/clean_pipeline.py
输入: data/raw/qianniu_messages.jsonl, satisfaction.jsonl, orders.jsonl
输出: data/clean/sft_train.jsonl, data/clean/rejected.jsonl, 终端打印漏斗统计
"""

import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "clean"

TS_FORMAT = "%Y-%m-%d %H:%M:%S"
SESSION_GAP_MINUTES = 30          # 同一买家超过该间隔算两个会话
TARGET_RATIO = {"售前": 0.6, "售后": 0.3, "安抚": 0.1}
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "你是某国产美妆品牌天猫店的客服，称呼客户为\"宝子\"，语气亲昵活泼，"
    "每条回复 emoji 不超过 3 个。价格、库存、券、物流等实时事实由上下文注入，"
    "不得自行编造数字。过敏等高危场景一律安抚后转人工。"
    "禁止使用美白、祛斑、治疗等违规宣称。"
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1. 拆会话: 按时间间隔(30 分钟)把同一买家的消息流水切成 session
# ---------------------------------------------------------------------------
def split_sessions(messages, gap_minutes=SESSION_GAP_MINUTES):
    """按时间间隔拆分会话"""
    sessions = []
    current = [messages[0]]
    for i in range(1, len(messages)):
        gap = (messages[i]["dt"] - messages[i - 1]["dt"]).total_seconds() / 60
        if gap > gap_minutes:
            sessions.append(current)
            current = [messages[i]]
        else:
            current.append(messages[i])
    sessions.append(current)
    return sessions


# ---------------------------------------------------------------------------
# 2. 脱敏: 手机号/地址/订单号/买家昵称 -> 占位符
# ---------------------------------------------------------------------------
DESENSITIZE_RULES = [
    ("手机号", re.compile(r"1[3-9]\d{9}"), "[手机号]"),
    ("订单号", re.compile(r"TB\d{12,}"), "[订单号]"),
    ("地址", re.compile(r"[\u4e00-\u9fa5]{2,}省[\u4e00-\u9fa5]{2,}市[\u4e00-\u9fa5]{2,}(?:区|县)[^，。,]{2,30}(?:室|号|楼|单元)"), "[地址]"),
]


def desensitize(text, buyer, counter):
    for name, pattern, placeholder in DESENSITIZE_RULES:
        text, n = pattern.subn(placeholder, text)
        counter[name] += n
    # 买家昵称出现在消息正文里(如收货人署名)也要替换
    if buyer in text:
        counter["买家昵称"] += text.count(buyer)
        text = text.replace(buyer, "[买家昵称]")
    return text


# ---------------------------------------------------------------------------
# 4. 质量筛选
# ---------------------------------------------------------------------------
COMPLAINT_PATTERN = re.compile(r"垃圾|投诉|再也不|欺骗|必须.*(?:说法|负责)")

# 演示用判别规则(生产环境该位置是 LLM 判别)
BANNED_PATTERN = re.compile(r"美白|祛斑|治疗|垃圾|亲[，,～~]")
POLITICS_PATTERN = re.compile(r"政策|热搜|新闻")
REFUSAL_PATTERN = re.compile(r"不用了|算了|太贵")


def keyword_judge(session):
    """无星无成交会话的质量判别(演示版, 生产环境替换为 LLM 判别)"""
    msgs = session["messages"]
    service_msgs = [m["content"] for m in msgs if m["sender"] == "service"]
    if msgs[-1]["sender"] != "service":
        return False, "判别:会话未闭环"
    if any(BANNED_PATTERN.search(m) or POLITICS_PATTERN.search(m) for m in service_msgs):
        return False, "判别:违规或敏感内容"
    last_customer = [m["content"] for m in msgs if m["sender"] == "customer"][-1]
    if REFUSAL_PATTERN.search(last_customer):
        return False, "判别:客户未接受"
    if sum(len(m) for m in service_msgs) / len(service_msgs) < 12:
        return False, "判别:回复无实质内容"
    return True, "判别:优质"


def is_quality_session(session):
    """返回 (是否入选, 原因)。基础门槛 -> 打星快速通道 -> 多信号通道"""
    msgs = session["messages"]
    # 基础门槛: 所有会话都要过
    if len(msgs) < 6:  # 至少 3 轮对话
        return False, "基础:不足3轮"
    last = msgs[-1]
    if last["sender"] == "customer" and COMPLAINT_PATTERN.search(last["content"]):
        return False, "基础:结尾客户抱怨"
    service_msgs = [m["content"] for m in msgs if m["sender"] == "service"]
    if all(len(m) < 5 for m in service_msgs):
        return False, "基础:客服敷衍短句"

    # 快速通道: 有打分的, >=4 星直接入选
    if session["rating"] is not None:
        if session["rating"] >= 4:
            return True, f"快速通道:{session['rating']}星"
        return False, f"快速通道:低星{session['rating']}"

    # 多信号通道: 无打分的, 靠成交信号或 LLM 判别
    if session["ordered_within_24h"]:
        return True, "多信号:24h内成交"
    return keyword_judge(session)


# ---------------------------------------------------------------------------
# 5. 场景分类(演示版, 生产环境替换为 LLM 批量打标)
# ---------------------------------------------------------------------------
COMFORT_PATTERN = re.compile(r"累|烦|加班|空瓶|太好用|心情不好")
AFTER_SALE_PATTERN = re.compile(
    r"快递|物流|发货|签收|收到|退|换|少发|漏发|碎|破损|发票|价保|差价|地址|搓泥|过敏|红又痒|说法"
)


def classify_session(session):
    text = "".join(m["content"] for m in session["messages"] if m["sender"] == "customer")
    if COMFORT_PATTERN.search(text):
        return "安抚"
    if AFTER_SALE_PATTERN.search(text):
        return "售后"
    return "售前"


# ---------------------------------------------------------------------------
# 6. 事实占位符化: 易变事实不进模型记忆, 在线链路由接口实时注入
# ---------------------------------------------------------------------------
def mask_facts(text):
    # 规则有先后顺序: 先命中组合形态, 再兜底单个价格
    rules = [
        (re.compile(r"满\d+减\d+"), "[满减券]"),
        (re.compile(r"原价\d+(?:\.\d+)?元?"), "原价[原价]"),
        (re.compile(r"加起来要\d+(?:\.\d+)?元"), "加起来要[原价]"),
        (re.compile(r"活动价\d+(?:\.\d+)?元?"), "活动价[当前活动价]"),
        (re.compile(r"\d+(?:\.\d+)?%烟酰胺"), "[成分浓度]烟酰胺"),
        (re.compile(r"保质期\d+年"), "保质期[保质期]"),
        (re.compile(r"\d+个月内用完"), "[开封后使用期]内用完"),
        (re.compile(r"已到[\u4e00-\u9fa5]{2,12}(?:中心|站点|分部)"), "已到[最新物流节点]"),
        (re.compile(r"预计(?:明天|后天|\d+天)送达"), "预计[预计送达时间]送达"),
        (re.compile(r"\d+(?:\.\d+)?元"), "[当前活动价]"),
    ]
    for pattern, repl in rules:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    # 读取原始数据
    messages = load_jsonl(RAW_DIR / "qianniu_messages.jsonl")
    ratings = {(r["buyer"], r["date"]): r["rating"] for r in load_jsonl(RAW_DIR / "satisfaction.jsonl")}
    orders = load_jsonl(RAW_DIR / "orders.jsonl")
    for m in messages:
        m["dt"] = datetime.strptime(m["ts"], TS_FORMAT)
    for o in orders:
        o["dt"] = datetime.strptime(o["ts"], TS_FORMAT)

    # 1. 按买家分组拆会话
    by_buyer = {}
    for m in messages:
        by_buyer.setdefault(m["buyer"], []).append(m)
    sessions = []
    for buyer, msgs in by_buyer.items():
        msgs.sort(key=lambda m: m["dt"])
        for s in split_sessions(msgs):
            sessions.append({
                "buyer": buyer,
                "start": s[0]["dt"],
                "end": s[-1]["dt"],
                "messages": [{"sender": m["sender"], "content": m["content"]} for m in s],
            })
    print(f"[1] 拆会话: {len(messages)} 条消息 -> {len(sessions)} 个会话(间隔>{SESSION_GAP_MINUTES}分钟)")

    # 2. 脱敏(入库前完成)
    mask_counter = Counter()
    for s in sessions:
        for m in s["messages"]:
            m["content"] = desensitize(m["content"], s["buyer"], mask_counter)
    print(f"[2] 脱敏完成: {dict(mask_counter)}")

    # 3. 关联评分(按买家+日期)与成交信号(会话结束后 24h 内下单)
    for s in sessions:
        s["rating"] = ratings.get((s["buyer"], s["start"].date().isoformat()))
        s["ordered_within_24h"] = any(
            o["buyer"] == s["buyer"] and timedelta(0) <= o["dt"] - s["end"] <= timedelta(hours=24)
            for o in orders
        )
    rated = sum(1 for s in sessions if s["rating"] is not None)
    print(f"[3] 信号关联: 带评分 {rated} 个({rated / len(sessions):.0%}), 无评分 {len(sessions) - rated} 个")

    # 4. 质量筛选
    passed, rejected = [], []
    reason_counter = Counter()
    for s in sessions:
        ok, reason = is_quality_session(s)
        (passed if ok else rejected).append({**s, "reason": reason})
        reason_counter[reason] += 1
    print(f"[4] 质量筛选: 入选 {len(passed)} / {len(sessions)}")
    for reason, n in sorted(reason_counter.items()):
        print(f"      {reason}: {n}")

    # 5. 场景分类
    for s in passed:
        s["category"] = classify_session(s)
    dist = Counter(s["category"] for s in passed)
    print(f"[5] 场景分类: {dict(dist)}")

    # 6. 事实占位符化
    for s in passed:
        for m in s["messages"]:
            if m["sender"] == "service":
                m["content"] = mask_facts(m["content"])
    print("[6] 事实占位符化完成(价格/券/物流/保质期等 -> 占位符)")

    # 7. 配比调整: 售后与安抚全保留, 只对售前降采样, 对齐 60/30/10
    random.seed(RANDOM_SEED)
    kept = [s for s in passed if s["category"] in ("售后", "安抚")]
    presale = [s for s in passed if s["category"] == "售前"]
    # 以售后+安抚的量反推售前目标量: 售前 / (售后+安抚) = 60 / 40
    target_presale = round(len(kept) * TARGET_RATIO["售前"] / (TARGET_RATIO["售后"] + TARGET_RATIO["安抚"]))
    if len(presale) > target_presale:
        print(f"[7] 配比调整: 售前 {len(presale)} -> {target_presale}(降采样, 售后/安抚全保留)")
        presale = random.sample(presale, target_presale)
    else:
        print(f"[7] 配比调整: 售前 {len(presale)} 条未超目标 {target_presale}, 无需降采样")
    final = kept + presale

    # 8. 导出 SFT messages 格式
    out = []
    for i, s in enumerate(final, 1):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in s["messages"]:
            msgs.append({"role": "user" if m["sender"] == "customer" else "assistant", "content": m["content"]})
        out.append({"id": f"T{i:04d}", "category": s["category"], "messages": msgs})
    save_jsonl(out, OUT_DIR / "sft_train.jsonl")
    save_jsonl(
        [{"buyer": r["buyer"], "start": r["start"].strftime(TS_FORMAT), "reason": r["reason"],
          "messages": r["messages"]} for r in rejected],
        OUT_DIR / "rejected.jsonl",
    )
    final_dist = Counter(s["category"] for s in final)
    print(f"[8] 导出 {len(out)} 条 -> {OUT_DIR / 'sft_train.jsonl'}  最终配比: {dict(final_dist)}")
    print(f"    被淘汰 {len(rejected)} 条(含原因) -> {OUT_DIR / 'rejected.jsonl'}")


if __name__ == "__main__":
    main()
