# -*- coding: utf-8 -*-
"""演示界面支撑逻辑测试（R14）：MockGenerator + 采纳率统计 + 模块可导入。

对应 test_cases.md 用例：T24。
Streamlit UI 本身不做端到端测试，这里保证：
- MockGenerator 忠实引用 facts（回复数字通过数字校验），无 facts 时按意图出话术模板；
- stats.calc_adoption_rate / load_jsonl 统计逻辑正确；
- web_app 模块可 import（streamlit 运行时只走 main()，模块级不执行 UI 代码）。
"""
import json

from src.serving import fact_apis
from src.serving.pipeline import OutputFactChecker
from src.serving.promo_knowledge import FactChecker
from src.serving.stats import ADOPTION_ACTIONS, calc_adoption_rate, load_jsonl
from src.serving.web_app import MockGenerator


# ==================== MockGenerator：有 facts 忠实引用 ====================

def test_mock_generator_price_facts_passes_check():
    """商品价格事实注入：回复含现价/原价/券，数字全部通过数字校验。"""
    facts = fact_apis.get_product_info("P001")  # 129/169/满99减10
    reply = MockGenerator().generate(system="s", user="这款多少钱", facts=facts)
    assert "129" in reply and "169" in reply and "满99减10" in reply
    passed, unmatched = OutputFactChecker.verify(reply, facts)
    assert passed and unmatched == []


def test_mock_generator_price_without_coupon():
    """无券商品（P003）：不出现券文案，数字仍全部合法。"""
    facts = fact_apis.get_product_info("P003")  # 69/89/无券
    reply = MockGenerator().generate(system="s", user="多少钱", facts=facts)
    assert "69" in reply and "领" not in reply
    passed, _ = OutputFactChecker.verify(reply, facts)
    assert passed


def test_mock_generator_logistics_facts_passes_check():
    """物流事实注入：回复含状态/节点/预计送达，"2~3 天"的数字通过校验。"""
    facts = fact_apis.get_logistics("O20260618002")  # 已揽收/预计 2~3 天送达
    reply = MockGenerator().generate(system="s", user="快递到哪了", facts=facts)
    assert "已揽收" in reply and "2" in reply
    passed, unmatched = OutputFactChecker.verify(reply, facts)
    assert passed and unmatched == []


def test_mock_generator_promo_clauses_passes_check():
    """大促规则条款注入：原样引用条款，条款事实校验（数字/时间）通过。"""
    clause = "一、双11预售定金 50 元，支付定金可抵 100 元尾款。"
    reply = MockGenerator().generate(system="s", user="定金怎么算", facts={"活动规则": clause})
    assert "50" in reply and "100" in reply
    passed, unmatched = FactChecker.verify(reply, [clause])
    assert passed and unmatched == []


# ==================== MockGenerator：无 facts 按意图出模板 ====================

def test_mock_generator_no_facts_product_template():
    """无 facts 且产品类问法：出产品咨询模板，品牌语气（称呼宝子）。"""
    reply = MockGenerator().generate(system="s", user="油皮适合用这款面霜吗", facts=None)
    assert "宝子" in reply and "肤质" in reply


def test_mock_generator_no_facts_comfort_template():
    """无 facts 且情绪类问法：出安抚模板。"""
    reply = MockGenerator().generate(system="s", user="气死了，等这么久", facts=None)
    assert "宝子" in reply and "抱歉" in reply


def test_mock_generator_no_facts_default_template():
    """无 facts 且无明显关键词：出通用开场模板，不含数字。"""
    reply = MockGenerator().generate(system="s", user="在吗", facts=None)
    assert "宝子" in reply
    assert OutputFactChecker.extract_numbers(reply) == []


# ==================== 采纳率统计 ====================

def test_calc_adoption_rate_empty():
    """无记录：总数 0、采纳率 0.0、三种动作计数均为 0。"""
    stats = calc_adoption_rate([])
    assert stats["total"] == 0
    assert stats["adoption_rate"] == 0.0
    assert stats["counts"] == {action: 0 for action in ADOPTION_ACTIONS}


def test_calc_adoption_rate_mixed():
    """混合记录：直接发送 + 修改后发送 计为采纳，弃用不计。"""
    records = [
        {"action": "直接发送"},
        {"action": "直接发送"},
        {"action": "修改后发送"},
        {"action": "弃用"},
    ]
    stats = calc_adoption_rate(records)
    assert stats["total"] == 4
    assert stats["counts"] == {"直接发送": 2, "修改后发送": 1, "弃用": 1}
    assert stats["adoption_rate"] == 0.75


def test_load_jsonl(tmp_path):
    """load_jsonl：缺失文件返回空列表；正常解析每行 JSON，空行跳过。"""
    assert load_jsonl(tmp_path / "missing.jsonl") == []
    path = tmp_path / "log.jsonl"
    path.write_text(
        json.dumps({"action": "直接发送"}, ensure_ascii=False) + "\n\n"
        + json.dumps({"action": "弃用"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = load_jsonl(path)
    assert [r["action"] for r in records] == ["直接发送", "弃用"]


# ==================== 模块可导入（streamlit 运行时不执行） ====================

def test_web_app_module_importable():
    """import web_app 不触发 streamlit UI 运行时，main/MockGenerator/常量可用。"""
    import src.serving.web_app as web_app

    assert callable(web_app.main)
    assert web_app.MockGenerator is MockGenerator
    assert web_app.ADOPTION_LOG.name == "adoption_log.jsonl"
    assert len(web_app.PRESET_QUESTIONS) == 5
