# -*- coding: utf-8 -*-
"""T8 事实占位符化与训练集构建（对应 R6、R7）。

- 占位符化：只处理客服回复（sender=service），把价格/满减/库存等易变事实
  替换为占位符，映射规则留档 placeholder_mapping.json；
- 转 messages 格式：system 品牌人设 + 多轮 user/assistant；
- 分层抽样划分：按 category 分层，目标 train 800 / test 120，数据不够时
  按同一比例缩放并打印警告；划分结果无交集、可复现。

入口：python -m src.data_pipeline.build_dataset
"""
import json
import random
import re
from collections import Counter, defaultdict

from src.config import DATA_PROCESSED, DATA_TRAIN, SEED, TEST_SIZE, TRAIN_SIZE

# 事实占位符规则：(名称, 正则, 替换式)。满减等组合形态必须排在单个价格之前。
PLACEHOLDER_RULES = [
    ("优惠满减", re.compile(r"满\d+减\d+"), "[当前优惠]"),
    ("价格", re.compile(r"\d+(?:\.\d+)?元"), "[当前活动价]"),
    ("库存", re.compile(r"(库存|现货)\d+(?:件|个|支|瓶)?"), r"\g<1>[当前库存]"),
]

# system 品牌人设：身份、称呼、语气、事实注入约束、禁用语约束
SYSTEM_PROMPT = (
    "你是某国产美妆品牌天猫店的客服，称呼客户为\"宝子\"，语气亲昵活泼，"
    "每条回复 emoji 不超过 3 个。价格、库存、优惠等实时事实由上下文注入，"
    "不得自行编造数字，表达时统一使用占位符。过敏等高危场景一律安抚后转人工。"
    "禁止使用美白、祛斑、治疗等违规宣称。"
)


def placeholderize(text):
    """把客服回复中的价格/满减/库存数字替换为占位符。"""
    for _, pattern, repl in PLACEHOLDER_RULES:
        text = pattern.sub(repl, text)
    return text


def to_sft_sample(session):
    """会话 -> {"id", "category", "messages"}；只掩码客服回复，客户消息保持原样。"""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in session["messages"]:
        if m["sender"] == "service":
            msgs.append({"role": "assistant", "content": placeholderize(m["text"])})
        else:
            msgs.append({"role": "user", "content": m["text"]})
    return {"id": session["session_id"], "category": session["category"], "messages": msgs}


def stratified_split(samples, train_size=TRAIN_SIZE, test_size=TEST_SIZE, seed=SEED):
    """按 category 分层抽样划分训练/测试集，各类别内部按同一比例切分，无交集。"""
    if len(samples) < train_size + test_size:
        print(f"  警告: 样本总数 {len(samples)} 不足目标 {train_size}+{test_size}，"
              f"按同一比例缩放到现有数据量")
    test_frac = test_size / (train_size + test_size)
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for s in samples:
        by_cat[s["category"]].append(s)
    train, test = [], []
    for cat in sorted(by_cat):
        items = by_cat[cat]
        rng.shuffle(items)
        # 每类至少留 1 条进训练集
        n_test = min(round(len(items) * test_frac), len(items) - 1)
        n_test = max(n_test, 0)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _dist_str(samples):
    dist = Counter(s["category"] for s in samples)
    total = len(samples)
    return {k: f"{v}（{v / total:.1%}）" for k, v in sorted(dist.items())} if total else {}


def main():
    src = DATA_PROCESSED / "sessions_labeled.jsonl"
    DATA_TRAIN.mkdir(parents=True, exist_ok=True)
    train_path = DATA_TRAIN / "train.jsonl"
    test_path = DATA_TRAIN / "test.jsonl"
    mapping_path = DATA_PROCESSED / "placeholder_mapping.json"

    with open(src, encoding="utf-8") as f:
        sessions = [json.loads(line) for line in f if line.strip()]

    samples = [to_sft_sample(s) for s in sessions]
    train, test = stratified_split(samples)

    for path, rows in ((train_path, train), (test_path, test)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 占位符映射规则留档
    mapping = [
        {"name": name, "pattern": p.pattern, "replacement": repl}
        for name, p, repl in PLACEHOLDER_RULES
    ]
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[T8] 训练集构建完成: 训练集 {len(train)} 条 / 测试集 {len(test)} 条（无交集）")
    print(f"  训练集分布: {_dist_str(train)}")
    print(f"  测试集分布: {_dist_str(test)}")
    print(f"  输出: {train_path}")
    print(f"  输出: {test_path}")
    print(f"  映射留档: {mapping_path}")


if __name__ == "__main__":
    main()
