# -*- coding: utf-8 -*-
"""BLEU 评估（R15 / T25）：SFT 前后趋势对比，三层评估中的自动参考层。

取舍说明：中文按字符切分而不引入 jieba 分词——BLEU 在本项目只看 SFT 前后
的相对趋势（叙事中 0.15 -> 0.25），不看绝对值；字切分对两组候选影响一致，
不破坏趋势可比性，且免增依赖。

候选回复来源抽象为函数参数 / 文件输入，不直接耦合 RemoteGenerator：
调用方先用推理服务把测试集首问跑成生成结果文件，再交给本模块算分，
便于桩数据测试，也便于后续替换真实生成来源。

入口：
    python -m src.evaluation.bleu_eval --candidates <生成结果.jsonl> \
        [--base-candidates <通用模型结果.jsonl>] [--test-file <测试集>]

约定：候选文件与测试集逐行一一对应（每行一条 reply，对应测试集同序号样本的
首条客户问题）；参考文本取测试集该样本的首条 assistant 真实回复。
"""
import argparse
import json
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu

from src.config import DATA_PROCESSED, DATA_TRAIN
from src.evaluation.compliance import load_replies


def tokenize(text: str) -> list[str]:
    """中文字符级切分：去掉空白后逐字作为 token。"""
    return [ch for ch in text if not ch.isspace()]


def calc_bleu(references: list[str], candidates: list[str]) -> float:
    """批量计算平均 sentence BLEU，references 与 candidates 等长一一对应。"""
    if len(references) != len(candidates):
        raise ValueError(
            f"参考与候选数量不一致: {len(references)} vs {len(candidates)}"
        )
    scores = [
        sentence_bleu([tokenize(ref)], tokenize(cand))
        for ref, cand in zip(references, candidates)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def extract_reference(sample: dict) -> str:
    """取测试样本的首条 assistant 真实回复作为参考文本。"""
    return next(m["content"] for m in sample["messages"] if m["role"] == "assistant")


def main() -> None:
    parser = argparse.ArgumentParser(description="BLEU 趋势对比（R15 / T25）")
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="SFT 模型生成结果 jsonl（每行一条 reply，与测试集逐行对应）",
    )
    parser.add_argument(
        "--base-candidates",
        type=Path,
        default=None,
        help="通用模型生成结果 jsonl（可选，传入则输出 SFT 前后趋势）",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DATA_TRAIN / "test.jsonl",
        help="测试集路径，默认 data/train/test.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED / "bleu_report.json",
        help="报告输出路径",
    )
    args = parser.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    references = [extract_reference(s) for s in samples]

    candidates = load_replies(args.candidates)
    bleu = calc_bleu(references, candidates)
    report = {"count": len(references), "bleu": bleu}
    print(f"[SFT 模型] BLEU = {bleu:.4f}（{len(references)} 条）")

    if args.base_candidates:
        base_bleu = calc_bleu(references, load_replies(args.base_candidates))
        report["base_bleu"] = base_bleu
        print(f"[通用模型] BLEU = {base_bleu:.4f}")
        print(f"[趋势] BLEU {base_bleu:.4f} -> {bleu:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已输出: {args.output}")


if __name__ == "__main__":
    main()
