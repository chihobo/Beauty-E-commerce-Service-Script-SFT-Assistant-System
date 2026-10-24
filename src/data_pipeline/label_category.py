# -*- coding: utf-8 -*-
"""T7 会话分类与配比控制（对应 R5）。

- classify_session(session_text, client)：LLM 把会话分类到 售前/售后/闲聊安抚，
  依赖注入客户端；返回无法识别时直接报错，不做兜底；
- rebalance(sessions, target_ratio)：配比调整——售前降采样、售后/闲聊全保留，
  以售后+闲聊的量反推售前目标量（售前 :（售后+闲聊） = 60 : 40）。

入口：python -m src.data_pipeline.label_category
"""
import json
import random
from collections import Counter

from src.config import CATEGORY_RATIO, DATA_PROCESSED, LLM_MODEL, SEED, get_llm_client
from src.data_pipeline.filter_quality import session_to_text

CATEGORIES = ["售前", "售后", "闲聊安抚"]

CLASSIFY_PROMPT = (
    "你是电商会话分类器。把下面的客服会话分类到三类之一：\n"
    "- 售前：商品咨询，如肤质匹配、成分、价格、优惠、色号等；\n"
    "- 售后：物流查询、退换货、使用方法、过敏投诉等；\n"
    "- 闲聊安抚：客户发泄情绪或闲聊，客服安抚陪伴。\n"
    "只回复类别名：售前 / 售后 / 闲聊安抚。\n\n会话内容：\n{text}"
)


def classify_session(session_text, client):
    """LLM 打标签，返回 售前/售后/闲聊安抚 之一。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(text=session_text)}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    for c in CATEGORIES:
        if c in answer:
            return c
    raise RuntimeError(f"LLM 分类返回无法识别（期望 {CATEGORIES} 之一）: {answer!r}")


def rebalance(sessions, target_ratio=CATEGORY_RATIO, seed=SEED):
    """配比调整：售前降采样，售后/闲聊安抚全保留。

    以售后+闲聊的量反推售前目标量：售前 /（售后+闲聊） = 售前配比 / 非售前配比；
    售前不足目标量时不做降采样，全量保留。
    """
    rng = random.Random(seed)
    presale = [s for s in sessions if s["category"] == "售前"]
    kept = [s for s in sessions if s["category"] != "售前"]
    pre_ratio = target_ratio["售前"]
    other_ratio = sum(v for k, v in target_ratio.items() if k != "售前")
    target_presale = round(len(kept) * pre_ratio / other_ratio)
    if len(presale) > target_presale:
        presale = rng.sample(presale, target_presale)
    result = kept + presale
    rng.shuffle(result)
    return result


def main(client=None):
    if client is None:
        client = get_llm_client()
    src = DATA_PROCESSED / "sessions_quality.jsonl"
    dst = DATA_PROCESSED / "sessions_labeled.jsonl"
    report_path = DATA_PROCESSED / "rebalance_report.json"

    with open(src, encoding="utf-8") as f:
        sessions = [json.loads(line) for line in f if line.strip()]

    # LLM 批量打标签
    for i, s in enumerate(sessions, 1):
        s["category"] = classify_session(session_to_text(s), client)
        if i % 50 == 0 or i == len(sessions):
            print(f"  打标进度: {i}/{len(sessions)}")
    before = Counter(s["category"] for s in sessions)

    # 配比调整
    rebalanced = rebalance(sessions)
    after = Counter(s["category"] for s in rebalanced)

    report = {
        "before": dict(before),
        "after": dict(after),
        "target_ratio": CATEGORY_RATIO,
    }
    with open(dst, "w", encoding="utf-8") as f:
        for s in rebalanced:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total_after = sum(after.values())
    print(f"[T7] 分类与配比调整完成")
    print(f"  调整前分布: {dict(before)}")
    print(f"  调整后分布: {dict(after)}（合计 {total_after}，目标配比 {CATEGORY_RATIO}）")
    print(f"  输出: {dst}")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
