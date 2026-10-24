# -*- coding: utf-8 -*-
"""三层评估测试（对应 T25、T26、T27 / R15）。

全部使用临时构造的假数据，不依赖真实 test.jsonl 与真实推理服务。
"""
from src.evaluation.bleu_eval import calc_bleu
from src.evaluation.blind_test import build_blind_test
from src.evaluation.compliance import batch_check, check_compliance


# ---------- 合规率检查（T26） ----------

def test_banned_word_hit():
    ok, violations = check_compliance("宝子，这款美白祛斑效果超好")
    assert not ok
    assert any("美白" in v for v in violations)
    assert any("祛斑" in v for v in violations)


def test_qin_address_violation():
    """用了"亲"且没叫"宝子"，判称呼违规。"""
    ok, violations = check_compliance("亲，这款很不错的")
    assert not ok
    assert any("称呼不规范" in v for v in violations)


def test_baozi_address_pass():
    ok, violations = check_compliance("宝子，这款很不错的～")
    assert ok
    assert violations == []


def test_emoji_boundary():
    """3 个 emoji 合规，4 个违规。"""
    ok3, _ = check_compliance("宝子这款不错😊😊😊")
    assert ok3
    ok4, violations4 = check_compliance("宝子这款不错😊😊😊😊")
    assert not ok4
    assert any("emoji 过多" in v for v in violations4)


def test_batch_check_rate():
    replies = [
        "宝子，这款很不错的～",          # 合规
        "亲，这款很不错的",              # 称呼违规
        "宝子，这款能祛斑哦",            # 禁用语违规
        "宝子，好的😊😊😊😊",            # emoji 违规
    ]
    result = batch_check(replies)
    assert result["total"] == 4
    assert result["passed"] == 1
    assert result["compliance_rate"] == 0.25
    # 三条违规明细，索引与原列表对应
    assert [v["index"] for v in result["violations"]] == [1, 2, 3]


# ---------- BLEU 评估（T25） ----------

def test_bleu_identical():
    assert calc_bleu(["宝子这款现在活动价"], ["宝子这款现在活动价"]) == 1.0


def test_bleu_disjoint_near_zero():
    assert calc_bleu(["宝子这款现在活动价"], ["天地玄黄宇宙洪荒"]) == 0.0


def test_bleu_batch_mean():
    score = calc_bleu(
        ["宝子这款现在活动价", "天地玄黄宇宙洪荒"],
        ["宝子这款现在活动价", "abcdefghijklmnopqrstuvwxyz你好"],
    )
    # 第一条完全相同 1.0，第二条完全不同 0.0，均值 0.5
    assert score == 0.5


# ---------- 盲测问卷（T27） ----------

def _fake_pool(size):
    questions = [f"问题{i}：这款多少钱" for i in range(size)]
    sft = [f"SFT 回复{i}：宝子活动价[当前活动价]" for i in range(size)]
    generic = [f"通用回复{i}：您好，很高兴为您解答" for i in range(size)]
    human = [f"人工回复{i}：宝子拍下今天发" for i in range(size)]
    return questions, sft, generic, human


def test_blind_sample_count():
    """150 条池子抽 100 题；不足 100 条时全量出题。"""
    md, key = build_blind_test(*_fake_pool(150), n=100)
    assert len(key) == 100
    assert "第 100 题" in md

    md_small, key_small = build_blind_test(*_fake_pool(3), n=100)
    assert len(key_small) == 3
    assert "第 3 题" in md_small


def test_blind_options_complete_and_key_recovers():
    """每题三个选项恰好是 SFT/通用/真人三答案，映射可还原来源与文本。"""
    questions, sft, generic, human = _fake_pool(10)
    _, key = build_blind_test(questions, sft, generic, human, n=10)
    assert len(key) == 10
    for item in key:
        idx = item["sample_index"]
        sources = {opt["source"] for opt in item["options"].values()}
        assert sources == {"sft", "generic", "human"}
        texts = {opt["text"] for opt in item["options"].values()}
        assert texts == {sft[idx], generic[idx], human[idx]}
        # correct 指向 SFT 答案所在选项
        assert item["options"][item["correct"]]["source"] == "sft"
        assert item["options"][item["correct"]]["text"] == sft[idx]


def test_blind_questionnaire_no_source_leak():
    """问卷正文不得出现来源字样（SFT/真人/通用/人工）。"""
    # 构造答案文本本身不含来源词，检验问卷模板不额外泄漏
    questions = [f"问题{i}" for i in range(5)]
    replies = [f"宝子好的{i}" for i in range(5)]
    md, _ = build_blind_test(questions, replies, replies, replies, n=5)
    for word in ("SFT", "真人", "通用", "人工"):
        assert word not in md


def test_blind_reproducible():
    pool = _fake_pool(20)
    _, key1 = build_blind_test(*pool, n=10)
    _, key2 = build_blind_test(*pool, n=10)
    assert key1 == key2
