# -*- coding: utf-8 -*-
"""T4 脱敏：手机号/地址/订单号/买家昵称 -> 占位符（对应 R2）。

- 脱敏规则集中在 DESENSITIZE_RULES（名称/正则/占位符），可配置；
- 买家昵称不是正则能覆盖的，按 customer_id -> nickname 映射做文本替换；
- 产出脱敏报告 desensitize_report.json（各规则命中数）并打印前后对比样例；
- scan_sensitive() 为全量扫描校验函数：脱敏后的数据应扫不出任何敏感信息。

入口：python -m src.data_pipeline.desensitize
"""
import json
import re
from collections import Counter

from src.config import DATA_PROCESSED, DATA_RAW

# 脱敏规则：(名称, 编译后的正则, 占位符)。手机号必须在订单号之前替换，
# 避免 11 位手机号先被 \d{12,} 之外的方式误处理；地址只匹配"省+市+区/县+路号室"
# 完整形态，宁可漏掉不规范写法也不误伤正常对话。
DESENSITIZE_RULES = [
    ("手机号", re.compile(r"1[3-9]\d{9}"), "[手机号]"),
    ("订单号", re.compile(r"\d{12,}"), "[订单号]"),
    ("地址", re.compile(
        r"[一-龥]{2,}省[一-龥]{2,}市[一-龥]{2,}(?:区|县)[^，。,；;]{2,30}(?:室|号|楼|单元)"
    ), "[地址]"),
]
NICKNAME_RULE_NAME = "买家昵称"
NICKNAME_PLACEHOLDER = "[买家昵称]"


def desensitize_text(text, nickname=None, counter=None):
    """对单条文本脱敏。counter 传入时累计各规则命中数。"""
    for name, pattern, placeholder in DESENSITIZE_RULES:
        text, n = pattern.subn(placeholder, text)
        if counter is not None:
            counter[name] += n
    # 买家昵称出现在正文里（如收货人署名）也要替换
    if nickname and nickname in text:
        if counter is not None:
            counter[NICKNAME_RULE_NAME] += text.count(nickname)
        text = text.replace(nickname, NICKNAME_PLACEHOLDER)
    return text


def scan_sensitive(text, nicknames=None):
    """扫描文本中残留的敏感信息，返回命中的规则名列表（应为空）。"""
    hits = [name for name, pattern, _ in DESENSITIZE_RULES if pattern.search(text)]
    if nicknames and any(n in text for n in nicknames):
        hits.append(NICKNAME_RULE_NAME)
    return hits


def main():
    src = DATA_RAW / "messages.jsonl"
    dst = DATA_PROCESSED / "messages_desensitized.jsonl"
    report_path = DATA_PROCESSED / "desensitize_report.json"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    counter = Counter()
    samples = []  # 前后对比样例（最多留 3 条发生变化的）
    total = 0
    changed = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            m = json.loads(line)
            before = m["text"]
            m["text"] = desensitize_text(before, nickname=m["nickname"], counter=counter)
            total += 1
            if m["text"] != before:
                changed += 1
                if len(samples) < 3:
                    samples.append({"before": before, "after": m["text"]})
            fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    report = {"total_messages": total, "messages_changed": changed, "hits": dict(counter)}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[T4] 脱敏完成: {total} 条消息, {changed} 条发生替换, 各规则命中: {dict(counter)}")
    for s in samples:
        print(f"  前: {s['before']}")
        print(f"  后: {s['after']}")
    print(f"  输出: {dst}")
    print(f"  报告: {report_path}")


if __name__ == "__main__":
    main()
