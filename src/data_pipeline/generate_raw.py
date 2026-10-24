# -*- coding: utf-8 -*-
"""T3 数据生成：模拟千牛消息流水生成器（对应 R1）。

产出两个文件（data/raw/）：
- messages.jsonl      消息流水，每行一条消息：
                      {timestamp(秒), customer_id, nickname, sender(customer/service), text}
- sessions_meta.json  会话级元信息（打星 rating 1-5 或 None、24h 成交标记等），
                      以 "{customer_id}#{序号:02d}" 为键（序号为该客户会话按时间先后的编号），
                      T5 拆会话时按同一规则重建 session_id 并把元信息挂回。

设计说明：
- 元信息独立存放而不嵌入消息流，是为了还原真实项目"评分表/订单表独立、需关联回会话"的形态；
- meta 中 scenario / quality_tier / has_sensitive 为 demo 自检字段（真实项目不存在），
  只用于生成后打印统计核对，不进入后续管线；
- 分布设定：售前约 68% / 售后 24% / 闲聊 8%（原始未配比，售后/闲聊给足绝对数量供配比锚定）；打星覆盖率约 30% 且 J 型
  （5 星为主、1 星次之）；约 20% 会话含敏感信息；约 25% 无星会话带 24h 成交标记；
  另掺入约 8% 垃圾会话（短会话/结尾抱怨/敷衍短句）用于演示基础门槛过滤；
- 固定随机种子，输出可复现。

入口：python -m src.data_pipeline.generate_raw
"""
import json
import random
from collections import Counter
from datetime import datetime, timedelta

from src.config import DATA_RAW, SEED
from src.data_pipeline import templates as T

# ---------- 生成规模与分布常量 ----------
NUM_CUSTOMERS = 800          # 客户数（期望产出约 9000 条消息流水）
BASE_DT = datetime(2025, 3, 1, 9, 0, 0)  # 消息起始基准时间
SPAN_DAYS = 14               # 消息流水跨度（天）

SESSIONS_PER_CUSTOMER = (1, 2, 3)
SESSIONS_WEIGHTS = (0.70, 0.22, 0.08)     # 同一客户的会话数分布

# 场景分布：售后/闲聊是配比锚定项（全保留），原始分布给足绝对数量，
# 售前靠 T7 降采样压回 60%，保证最终训练集规模
SCENARIO_WEIGHTS = {"售前": 0.68, "售后": 0.24, "闲聊安抚": 0.08}
JUNK_RATIO = 0.08                         # 垃圾会话占比（演示基础门槛）
RATING_COVER = 0.30                       # 正常会话打星覆盖率（J 型见 RATING_WEIGHTS）
RATING_WEIGHTS = {5: 0.70, 4: 0.10, 3: 0.05, 2: 0.03, 1: 0.12}
TIER_WEIGHTS = {T.TIER_GOLD: 0.60, T.TIER_ROOKIE: 0.25, T.TIER_OUTSOURCED: 0.15}
SENSITIVE_RATIO = 0.20                    # 正常会话含敏感信息比例
ORDERED_RATIO_UNRATED = 0.25              # 无星会话带 24h 成交标记比例

_SCENARIO_TEMPLATES = {
    "售前": T.PRE_SALE_TEMPLATES,
    "售后": T.AFTER_SALE_TEMPLATES,
    "闲聊安抚": T.CHAT_TEMPLATES,
}


def _pick(rng, weights: dict):
    """按权重字典抽一个键。"""
    return rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]


def _build_pairs(rng, tier):
    """选一个正常模板并按质量分层展开成 (sender, text) 列表，返回 (pairs, scenario, has_sensitive)。"""
    scenario = _pick(rng, SCENARIO_WEIGHTS)
    tpl = rng.choice(_SCENARIO_TEMPLATES[scenario])
    pairs = []
    for turn in tpl["turns"]:
        pairs.append(("customer", turn["customer"]))
        pairs.append(("service", turn["service"][tier]))
    has_sensitive = rng.random() < SENSITIVE_RATIO
    if has_sensitive:
        snippet = rng.choice(T.SENSITIVE_SNIPPETS)
        # 客户消息中的 {nickname} 是昵称引用占位，生成期替换为具体昵称（在调用方完成）
        pairs.insert(2, ("customer", snippet["customer"]))
        pairs.insert(3, ("service", snippet["ack"][tier]))
    return pairs, scenario, has_sensitive


def _assign_timestamps(rng, pairs, start_dt):
    """给 (sender, text) 列表按 20s~150s 间隔顺序分配时间戳。"""
    t = start_dt
    timed = []
    for sender, text in pairs:
        timed.append((int(t.timestamp()), sender, text))
        t += timedelta(seconds=rng.randint(20, 150))
    return timed


def generate_corpus(num_customers=NUM_CUSTOMERS, seed=SEED):
    """生成模拟消息流水与会话元信息。

    返回 (messages, sessions_meta)：
    - messages: 按时间戳排序的消息字典列表；
    - sessions_meta: session_id -> {rating, ordered_within_24h, scenario,
      quality_tier, has_sensitive}。
    """
    rng = random.Random(seed)
    messages = []
    meta = {}

    for ci in range(num_customers):
        customer_id = f"buyer_{ci:04d}"
        nickname = T.NICKNAME_POOL[ci % len(T.NICKNAME_POOL)]
        n_sessions = rng.choices(SESSIONS_PER_CUSTOMER, weights=SESSIONS_WEIGHTS, k=1)[0]
        # 同一客户的多个会话分布在不同日期，保证间隔远超 30 分钟拆分阈值
        days = sorted(rng.sample(range(SPAN_DAYS), n_sessions))
        for si, day in enumerate(days):
            session_id = f"{customer_id}#{si:02d}"
            tier = _pick(rng, TIER_WEIGHTS)
            start_dt = BASE_DT + timedelta(
                days=day, hours=rng.randint(0, 12), minutes=rng.randint(0, 59)
            )
            if rng.random() < JUNK_RATIO:
                tpl = rng.choice(T.JUNK_TEMPLATES)
                pairs = list(tpl["messages"])
                scenario, has_sensitive = tpl["scenario"], False
                rating, ordered = None, False
            else:
                pairs, scenario, has_sensitive = _build_pairs(rng, tier)
                rating = (
                    _pick(rng, RATING_WEIGHTS) if rng.random() < RATING_COVER else None
                )
                ordered = rating is None and rng.random() < ORDERED_RATIO_UNRATED
            for ts, sender, text in _assign_timestamps(rng, pairs, start_dt):
                messages.append({
                    "timestamp": ts,
                    "customer_id": customer_id,
                    "nickname": nickname,
                    "sender": sender,
                    "text": text.format(nickname=nickname),
                })
            meta[session_id] = {
                "rating": rating,
                "ordered_within_24h": ordered,
                "scenario": scenario,
                "quality_tier": tier,
                "has_sensitive": has_sensitive,
            }

    messages.sort(key=lambda m: m["timestamp"])
    return messages, meta


def _print_stats(messages, meta):
    """生成后统计自检打印。"""
    dist = Counter(v["scenario"] for v in meta.values())
    ratings = [v["rating"] for v in meta.values()]
    rated = [r for r in ratings if r is not None]
    star_dist = Counter(rated)
    sensitive = sum(1 for v in meta.values() if v["has_sensitive"])
    unrated = [v for v in meta.values() if v["rating"] is None]
    ordered = sum(1 for v in unrated if v["ordered_within_24h"])
    print("[T3] 模拟千牛消息流水生成完成")
    print(f"  消息总数: {len(messages)} | 会话数: {len(meta)} | 客户数: {len({m['customer_id'] for m in messages})}")
    print(f"  场景分布: {dict(dist)}")
    print(f"  打星覆盖率: {len(rated) / len(meta):.1%}（星级分布: {dict(sorted(star_dist.items(), reverse=True))}）")
    print(f"  含敏感信息会话: {sensitive}（{sensitive / len(meta):.1%}）")
    print(f"  无星会话带 24h 成交标记: {ordered}/{len(unrated)}（{ordered / len(unrated):.1%}）")


def main():
    messages, meta = generate_corpus()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    msg_path = DATA_RAW / "messages.jsonl"
    meta_path = DATA_RAW / "sessions_meta.json"
    with open(msg_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _print_stats(messages, meta)
    print(f"  输出: {msg_path}")
    print(f"  输出: {meta_path}")


if __name__ == "__main__":
    main()
