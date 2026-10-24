# -*- coding: utf-8 -*-
"""合规率检查（R15 / T26）：话术规范自动批跑，三层评估中的验收核心指标。

检查项（与项目叙事一致，验收口径 72% -> 95%）：
1. 禁用语：化妆品广告法违规宣称词（BANNED_WORDS 集中可配置）；
2. 称呼规范：品牌要求统一称"宝子"，出现"亲"且未称"宝子"判违规；
3. emoji 上限：每条回复不超过 MAX_EMOJI 个。

入口：
    python -m src.evaluation.compliance --input <回复文件> [--base-input <通用模型回复文件>]

输入文件支持两种 jsonl 格式：
- 测试集格式（含 messages 字段）：提取其中全部 assistant 回复；
- 生成结果格式（含 reply 字段）：直接取 reply 文本。
"""
import argparse
import json
import re
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_TRAIN

# 禁用语词表：广告法违规宣称，集中在些可配置
BANNED_WORDS = ["美白", "祛斑", "治疗", "修复屏障"]
# 称呼规范：必须称"宝子"，"亲"为违规称呼
REQUIRED_ADDRESS = "宝子"
BANNED_ADDRESS = "亲"
# 每条回复 emoji 上限
MAX_EMOJI = 3
# emoji 码位范围：表情符号区（1F300-1F9FF）+ 杂项符号/装饰符号区（2600-27BF）
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001F9FF☀-➿]")


def check_compliance(reply: str) -> tuple[bool, list[str]]:
    """检查单条回复，返回（是否合规，违规明细列表）。"""
    violations = []
    for word in BANNED_WORDS:
        if word in reply:
            violations.append(f"禁用语: {word}")
    if BANNED_ADDRESS in reply and REQUIRED_ADDRESS not in reply:
        violations.append(
            f'称呼不规范: 用了"{BANNED_ADDRESS}"且未称"{REQUIRED_ADDRESS}"'
        )
    emoji_count = len(EMOJI_PATTERN.findall(reply))
    if emoji_count > MAX_EMOJI:
        violations.append(f"emoji 过多: {emoji_count} 个（上限 {MAX_EMOJI}）")
    return len(violations) == 0, violations


def batch_check(replies: list[str]) -> dict:
    """批量检查，返回合规率与违规明细。"""
    details = []
    passed = 0
    for idx, reply in enumerate(replies):
        ok, violations = check_compliance(reply)
        if ok:
            passed += 1
        else:
            details.append({"index": idx, "reply": reply, "violations": violations})
    total = len(replies)
    return {
        "total": total,
        "passed": passed,
        "compliance_rate": passed / total if total else 0.0,
        "violations": details,
    }


def load_replies(path: Path) -> list[str]:
    """从 jsonl 文件读取回复列表：测试集格式提取 assistant 回复，生成结果格式取 reply。"""
    replies = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                replies.extend(
                    m["content"] for m in obj["messages"] if m["role"] == "assistant"
                )
            else:
                replies.append(obj["reply"])
    return replies


def _print_summary(name: str, result: dict) -> None:
    """打印一组检查结果的摘要与违规类型分布。"""
    print(
        f"[{name}] 共 {result['total']} 条，合规 {result['passed']} 条，"
        f"合规率 {result['compliance_rate']:.1%}"
    )
    counter: dict[str, int] = {}
    for item in result["violations"]:
        for v in item["violations"]:
            key = v.split(":")[0]
            counter[key] = counter.get(key, 0) + 1
    for key, count in sorted(counter.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="合规率批跑（R15 / T26）")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_TRAIN / "test.jsonl",
        help="回复文件（测试集或生成结果 jsonl），默认测试集",
    )
    parser.add_argument(
        "--base-input",
        type=Path,
        default=None,
        help="通用模型回复文件（可选，传入则打印 SFT 前后对比）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED / "compliance_report.json",
        help="报告输出路径",
    )
    args = parser.parse_args()

    result = batch_check(load_replies(args.input))
    report = {"input": str(args.input), "sft": result}
    _print_summary("SFT 模型", result)

    if args.base_input:
        base_result = batch_check(load_replies(args.base_input))
        report["base"] = base_result
        _print_summary("通用模型", base_result)
        print(
            f"[对比] 合规率 {base_result['compliance_rate']:.1%} "
            f"-> {result['compliance_rate']:.1%}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已输出: {args.output}")


if __name__ == "__main__":
    main()
