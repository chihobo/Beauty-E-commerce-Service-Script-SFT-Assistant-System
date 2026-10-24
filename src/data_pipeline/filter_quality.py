# -*- coding: utf-8 -*-
"""T6 优质会话多信号筛选（对应 R4）。

分层漏斗（顺序不能乱）：
1. 基础门槛（所有会话必过）：>=6 条消息（3 轮）、最后一条不是客户抱怨、
   客服消息不全是 <5 字敷衍短句；
2. 快速通道：带星会话 rating >=4 直接入选，<4 直接淘汰；
3. 多信号通道：无星会话 ordered_within_24h 为真入选，否则由 LLM 判别裁决。

llm_judge(session_text, client) 通过依赖注入接 LLM 客户端，单元测试用桩 client，
不依赖真实 API；判别结果只接受 yes/no，其他返回直接报错（项目约定不做兜底）。

入口：python -m src.data_pipeline.filter_quality
"""
import json
from collections import Counter

from src.config import DATA_PROCESSED, LLM_MODEL, get_llm_client

# 客户抱怨词表（最后一条客户消息命中即淘汰）
COMPLAINT_WORDS = ["垃圾", "投诉", "再也不", "欺骗", "差评", "曝光"]

MIN_MESSAGES = 6  # 基础门槛：至少 3 轮对话（6 条消息）
MIN_SERVICE_TEXT_LEN = 5  # 客服回复低于该长度视为敷衍短句

JUDGE_PROMPT = (
    "你是一名电商客服质检员。请判断下面这段客服会话是否适合作为优质训练样本："
    "客服回复有实质内容、态度良好、无违规宣称（如美白/祛斑/治疗）、会话基本闭环。"
    "只回答 yes 或 no。\n\n会话内容：\n{text}"
)


def session_to_text(session):
    """把会话转成"客户：xxx / 客服：xxx"多行文本，供 LLM 阅读。"""
    lines = []
    for m in session["messages"]:
        role = "客户" if m["sender"] == "customer" else "客服"
        lines.append(f"{role}：{m['text']}")
    return "\n".join(lines)


def base_gate(session):
    """基础门槛。返回 (是否通过, 原因)。"""
    msgs = session["messages"]
    if len(msgs) < MIN_MESSAGES:
        return False, "基础:不足3轮"
    last = msgs[-1]
    if last["sender"] == "customer" and any(w in last["text"] for w in COMPLAINT_WORDS):
        return False, "基础:结尾客户抱怨"
    service_texts = [m["text"] for m in msgs if m["sender"] == "service"]
    if all(len(t) < MIN_SERVICE_TEXT_LEN for t in service_texts):
        return False, "基础:客服敷衍短句"
    return True, "基础:通过"


def llm_judge(session_text, client):
    """LLM 判别无星无成交会话的客服质量，只接受 yes/no 回答。"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(text=session_text)}],
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().lower()
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    raise RuntimeError(f"LLM 判别返回无法解析（期望 yes/no）: {answer!r}")


def filter_sessions(sessions, client):
    """分层筛选。返回 (入选会话列表, 漏斗报告 dict)。"""
    report = {
        "total": 0, "base_passed": 0, "base_rejected": 0,
        "fast_selected": 0, "fast_rejected": 0,
        "multi_selected": 0, "multi_rejected": 0,
        "final_selected": 0,
    }
    reasons = Counter()
    passed = []
    for s in sessions:
        report["total"] += 1
        ok, reason = base_gate(s)
        if not ok:
            report["base_rejected"] += 1
            reasons[reason] += 1
            continue
        report["base_passed"] += 1
        if s["rating"] is not None:
            # 快速通道：带星会话
            if s["rating"] >= 4:
                report["fast_selected"] += 1
                s["select_reason"] = f"快速通道:{s['rating']}星"
                passed.append(s)
            else:
                report["fast_rejected"] += 1
                reasons[f"快速通道:低星{s['rating']}"] += 1
        elif s["ordered_within_24h"]:
            # 多信号通道：24h 成交信号
            report["multi_selected"] += 1
            s["select_reason"] = "多信号:24h内成交"
            passed.append(s)
        elif llm_judge(session_to_text(s), client):
            # 多信号通道：LLM 判别
            report["multi_selected"] += 1
            s["select_reason"] = "多信号:LLM判别优质"
            passed.append(s)
        else:
            report["multi_rejected"] += 1
            reasons["多信号:LLM判别淘汰"] += 1
    report["final_selected"] = report["fast_selected"] + report["multi_selected"]
    report["reject_reasons"] = dict(reasons)
    return passed, report


def main(client=None):
    if client is None:
        client = get_llm_client()
    src = DATA_PROCESSED / "sessions.jsonl"
    dst = DATA_PROCESSED / "sessions_quality.jsonl"
    report_path = DATA_PROCESSED / "filter_report.json"

    with open(src, encoding="utf-8") as f:
        sessions = [json.loads(line) for line in f if line.strip()]

    passed, report = filter_sessions(sessions, client)

    with open(dst, "w", encoding="utf-8") as f:
        for s in passed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[T6] 优质筛选漏斗: 总数 {report['total']}"
          f" -> 过基础门槛 {report['base_passed']}（淘汰 {report['base_rejected']}）"
          f" -> 快速通道入选 {report['fast_selected']}（淘汰 {report['fast_rejected']}）"
          f" -> 多信号通道入选 {report['multi_selected']}（淘汰 {report['multi_rejected']}）"
          f" -> 最终入选 {report['final_selected']}")
    for reason, n in sorted(report["reject_reasons"].items()):
        print(f"    淘汰原因 {reason}: {n}")
    print(f"  输出: {dst}")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
