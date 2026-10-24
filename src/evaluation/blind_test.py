# -*- coding: utf-8 -*-
"""盲测问卷生成（R15 / T27）：三层评估中的人工拍板层。

流程（与项目叙事一致）：从测试集抽 N=100 条客户问题，每题配三条答案——
SFT 模型生成、通用模型生成、真人客服历史回复（取测试集 assistant 真实回复），
顺序随机打乱后产出问卷供店长盲选"哪个最像我们的人"；答案映射单独留档，
不写入问卷正文，避免来源泄漏。

入口：
    python -m src.evaluation.blind_test --sft <SFT结果.jsonl> --generic <通用结果.jsonl>

约定：两个生成结果文件与测试集逐行一一对应（每行一条 reply）。
产物：data/processed/blind_test_questionnaire.md（问卷）
     data/processed/blind_test_answer_key.json（答案映射留档）
"""
import argparse
import json
import random
from pathlib import Path

from src.config import DATA_PROCESSED, DATA_TRAIN, SEED
from src.evaluation.compliance import load_replies

# 盲测抽样题数（叙事：100 条交给店长盲选）
SAMPLE_SIZE = 100
# 三个答案来源标识：仅出现在答案映射留档中，问卷正文不得出现
SOURCE_SFT = "sft"
SOURCE_GENERIC = "generic"
SOURCE_HUMAN = "human"
OPTION_LETTERS = ("A", "B", "C")


def extract_qa(sample: dict) -> tuple[str, str]:
    """从测试样本提取首条客户问题与首条真人客服回复。"""
    msgs = sample["messages"]
    question = next(m["content"] for m in msgs if m["role"] == "user")
    human = next(m["content"] for m in msgs if m["role"] == "assistant")
    return question, human


def build_blind_test(
    questions: list[str],
    sft_replies: list[str],
    generic_replies: list[str],
    human_replies: list[str],
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> tuple[str, list[dict]]:
    """抽样并打乱三答案，返回（问卷 markdown，答案映射列表）。

    四个列表等长、逐条一一对应；不足 n 条时全量出题并打印提示。
    """
    total = len(questions)
    if not (len(sft_replies) == len(generic_replies) == len(human_replies) == total):
        raise ValueError(
            "四组数据数量不一致: "
            f"问题 {total} / SFT {len(sft_replies)} / "
            f"通用 {len(generic_replies)} / 真人 {len(human_replies)}"
        )

    rng = random.Random(seed)
    if total > n:
        indices = sorted(rng.sample(range(total), n))
    else:
        if total < n:
            print(f"[提示] 测试集仅 {total} 条，不足 {n} 条，全量出题")
        indices = list(range(total))

    lines = [
        "# 客服话术盲测问卷",
        "",
        "> 每题给出同一条客户咨询的 3 条回复，请勾选最像我们店客服的一条。",
        "",
    ]
    answer_key = []
    for qno, idx in enumerate(indices, start=1):
        # 每题在问卷中的选项顺序随机打乱，映射记录 字母 -> 来源
        answers = [
            (SOURCE_SFT, sft_replies[idx]),
            (SOURCE_GENERIC, generic_replies[idx]),
            (SOURCE_HUMAN, human_replies[idx]),
        ]
        rng.shuffle(answers)

        lines.append(f"## 第 {qno} 题")
        lines.append("")
        lines.append(f"客户：{questions[idx]}")
        lines.append("")
        options = {}
        correct = ""
        for letter, (source, text) in zip(OPTION_LETTERS, answers):
            lines.append(f"{letter}. {text}")
            options[letter] = {"source": source, "text": text}
            if source == SOURCE_SFT:
                correct = letter
        lines.append("")
        lines.append("勾选：____")
        lines.append("")
        answer_key.append(
            {
                "question": qno,
                "sample_index": idx,
                "options": options,
                "correct": correct,
            }
        )

    return "\n".join(lines), answer_key


def main() -> None:
    parser = argparse.ArgumentParser(description="盲测问卷生成（R15 / T27）")
    parser.add_argument(
        "--sft",
        type=Path,
        required=True,
        help="SFT 模型生成结果 jsonl（与测试集逐行对应）",
    )
    parser.add_argument(
        "--generic",
        type=Path,
        required=True,
        help="通用模型生成结果 jsonl（与测试集逐行对应）",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=DATA_TRAIN / "test.jsonl",
        help="测试集路径，默认 data/train/test.jsonl",
    )
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE, help="抽样题数")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PROCESSED,
        help="产物输出目录",
    )
    args = parser.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    qa = [extract_qa(s) for s in samples]
    questions = [q for q, _ in qa]
    human_replies = [h for _, h in qa]

    questionnaire, answer_key = build_blind_test(
        questions,
        load_replies(args.sft),
        load_replies(args.generic),
        human_replies,
        n=args.n,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    questionnaire_path = args.out_dir / "blind_test_questionnaire.md"
    answer_key_path = args.out_dir / "blind_test_answer_key.json"
    with open(questionnaire_path, "w", encoding="utf-8") as f:
        f.write(questionnaire)
    with open(answer_key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=2)

    print(f"[T27] 盲测问卷生成完成: 共 {len(answer_key)} 题")
    print(f"  问卷: {questionnaire_path}")
    print(f"  答案映射留档: {answer_key_path}")


if __name__ == "__main__":
    main()
