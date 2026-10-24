# -*- coding: utf-8 -*-
"""大促规则模块：规则文档上传 / 条款切块 / 有效期检索注入 / 事实校验 / 过期自动下线。

对应面试叙事：
- 大促前店长上传一页纸规则，按条款切块、标注有效期；
- 命中的条款注入生成上下文，回复里的数字/时间必须能在原文找到出处，否则拦截；
- 活动结束自动下线（检索层不可见）；
- 一年真正用上的就双 11、618 这种 2~4 次——常规券、直降这些"一句话优惠"
  在商品 API 返回里就有，走价格咨询意图，不动用这套机制。

检索取舍：规则文档本身只有一页纸，条款总字数通常远小于 1500，
此时全量注入比检索更稳（不会因相似度漏掉关键条款）；
只有总字数超限才退到 2-gram 字符相似度取 top3，刻意不上向量库。
"""
import json
import re
from datetime import date, datetime
from pathlib import Path

from src.config import DATA_PROCESSED

# 条款总字数低于该阈值时全量注入，超过才走相似度 top3（见模块 docstring 的取舍说明）
FULL_INJECTION_CHAR_LIMIT = 1500
# 相似度路径下返回的条款条数上限
TOP_K = 3

# 条款切块的编号模式：中文序号"一、二、"或阿拉伯序号"1. 2、"
_CLAUSE_HEAD = re.compile(r"(?m)^\s*(?:[一二三四五六七八九十]+、|\d+\s*[.、])")


class PromoKnowledgeStore:
    """大促规则文档存储：JSON 持久化，条款带有效期，过期自动下线。"""

    def __init__(self, store_path: Path | None = None):
        # 默认持久化到 data/processed/promo_docs.json，测试可传入临时路径
        self.store_path = Path(store_path) if store_path else DATA_PROCESSED / "promo_docs.json"
        self._docs = self._load()

    def _load(self) -> dict:
        if self.store_path.exists():
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _split_clauses(content: str) -> list[str]:
        """按条款编号模式切块：以"一、""1."等编号开头切分，保留编号在条款原文里。"""
        heads = [m for m in _CLAUSE_HEAD.finditer(content)]
        clauses = []
        for i, m in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(content)
            clause = content[m.start():end].strip()
            if clause:
                clauses.append(clause)
        return clauses

    @staticmethod
    def _to_date(d) -> date:
        """入参允许 date / datetime / 'YYYY-MM-DD' 字符串，统一归一为 date。"""
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        return date.fromisoformat(str(d))

    def upload_doc(self, title: str, content: str, start_date, end_date) -> list[str]:
        """上传规则文档：按条款切块并标注有效期，同名文档直接覆盖。

        返回切出的条款列表，供调用方确认切块结果。
        """
        clauses = self._split_clauses(content)
        self._docs[title] = {
            "title": title,
            "start_date": self._to_date(start_date).isoformat(),
            "end_date": self._to_date(end_date).isoformat(),
            "clauses": clauses,
        }
        self._save()
        return clauses

    def retrieve(self, query: str, now=None) -> list[str]:
        """检索有效期内的条款。

        过期 = 检索层不可见 = 自动下线。
        条款总字数 < FULL_INJECTION_CHAR_LIMIT 时全量返回（一页纸全量注入比检索更稳）；
        超过则用 2-gram 字符相似度对 query 与条款打分，取 top TOP_K。
        """
        now = self._to_date(now) if now else date.today()
        valid = [
            clause
            for doc in self._docs.values()
            if date.fromisoformat(doc["start_date"]) <= now <= date.fromisoformat(doc["end_date"])
            for clause in doc["clauses"]
        ]
        if not valid:
            return []
        if sum(len(c) for c in valid) < FULL_INJECTION_CHAR_LIMIT:
            return valid
        scored = sorted(
            valid,
            key=lambda c: self._bigram_similarity(query, c),
            reverse=True,
        )
        return scored[:TOP_K]

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        """归一化后取字符 2-gram 集合，用于相似度计算。"""
        normalized = FactChecker._normalize(text)
        if len(normalized) < 2:
            return {normalized} if normalized else set()
        return {normalized[i:i + 2] for i in range(len(normalized) - 1)}

    @classmethod
    def _bigram_similarity(cls, query: str, clause: str) -> float:
        """2-gram Jaccard 相似度；任一侧无 2-gram 时相似度为 0。"""
        q, c = cls._bigrams(query), cls._bigrams(clause)
        if not q or not c:
            return 0.0
        return len(q & c) / len(q | c)

    def cleanup_expired(self, now=None) -> list[str]:
        """物理清理过期条款（而非仅检索层隐藏），返回被清理的文档标题。

        条款有效期即文档有效期；文档整体过期则整篇删除。
        """
        now = self._to_date(now) if now else date.today()
        expired_titles = [
            title
            for title, doc in self._docs.items()
            if date.fromisoformat(doc["end_date"]) < now
        ]
        for title in expired_titles:
            del self._docs[title]
        if expired_titles:
            self._save()
        return expired_titles

    def handle_promo_query(self, query: str, reply_generator, now=None) -> dict:
        """大促规则链路：检索 -> 无有效条款转人工 -> 注入生成 -> 事实校验 -> 自动发送或拦截。

        reply_generator 为可调用对象 (query, clauses) -> str，依赖注入（测试用桩）：
        比如忠实引用条款的桩、编造"满200减30"的桩。

        返回决策字典：
        - decision: "auto_send"（校验通过）/ "draft"（编造被拦截转草稿）/ "human"（无有效条款转人工）
        - reply: 生成的回复（human 场景为 None）
        - clauses: 注入的条款
        - unmatched_facts: 未通过校验的事实（通过时为空列表）
        - reason: 决策依据，演示界面展示
        """
        clauses = self.retrieve(query, now=now)
        if not clauses:
            return {
                "decision": "human",
                "reply": None,
                "clauses": [],
                "unmatched_facts": [],
                "reason": "检索不到有效期内的活动规则条款，转人工处理",
            }
        reply = reply_generator(query, clauses)
        passed, unmatched = FactChecker.verify(reply, clauses)
        if passed:
            return {
                "decision": "auto_send",
                "reply": reply,
                "clauses": clauses,
                "unmatched_facts": [],
                "reason": "回复中的数字/时间事实均能在条款原文找到出处，自动发送",
            }
        return {
            "decision": "draft",
            "reply": reply,
            "clauses": clauses,
            "unmatched_facts": unmatched,
            "reason": f"回复含条款外的事实 {unmatched}，拦截转草稿",
        }


class FactChecker:
    """数字/时间事实校验：回复中的事实必须都能在条款原文找到出处。

    只校验阿拉伯数字（含时间点 20:00、日期 10月31日）。
    中文数字（"两件""三天"）刻意不校验：同一语义的中/阿文写法混用普遍
    （条款写"2 件"而回复写"两件"是忠实引用），硬校验误伤率太高，
    这类语义等价判断留给人工草稿环节。
    """

    # 时间点（20:00）与日期（10月31日）先于纯数字匹配，避免被拆碎
    _FACT_PATTERN = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*月\s*\d{1,2}\s*日|\d+(?:\.\d+)?")

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化：全角转半角、去空白，使"20：00"与"20:00"、"满 200 减 30"与"满200减30"等价。"""
        half_width = []
        for ch in text:
            code = ord(ch)
            if code == 0x3000:  # 全角空格
                continue
            if 0xFF01 <= code <= 0xFF5E:  # 全角符号转半角
                half_width.append(chr(code - 0xFEE0))
            else:
                half_width.append(ch)
        return re.sub(r"\s+", "", "".join(half_width))

    @classmethod
    def extract_facts(cls, text: str) -> list[str]:
        """从文本抽取阿拉伯数字事实（数字 / 时间点 / 日期），归一化后去重保序返回。"""
        facts = []
        for m in cls._FACT_PATTERN.finditer(text):
            fact = cls._normalize(m.group(0))
            if fact not in facts:
                facts.append(fact)
        return facts

    @classmethod
    def verify(cls, reply: str, source_clauses: list[str]) -> tuple[bool, list[str]]:
        """校验回复中的每个事实都能在条款原文中找到。

        条款原文同样用 extract_facts 抽成事实集合，回复事实必须精确命中集合，
        不做子串匹配——否则编造的"减 30"会被条款里的"满 300"误吞而漏拦截。

        返回 (是否通过, 未匹配的事实列表)。
        """
        source_facts = set(cls.extract_facts("".join(source_clauses)))
        unmatched = [
            fact for fact in cls.extract_facts(reply) if fact not in source_facts
        ]
        return (not unmatched, unmatched)
