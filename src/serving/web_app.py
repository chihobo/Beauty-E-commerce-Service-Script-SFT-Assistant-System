# -*- coding: utf-8 -*-
"""Streamlit 演示界面（R14）：面试现场演示主界面。

定位（对应面试叙事）：客服辅助工具，与千牛双窗口并排使用——
高置信且有事实接口的咨询由系统自动发送，其余全部进草稿，
由客服在界面上点「直接发送 / 修改后发送 / 弃用」。

页面结构：
- 侧边栏：意图分类版本切换（V1 规则 / V2 LLM，演示踩坑迭代故事）、
  生成器模式（本地模拟 / AutoDL 远程推理服务）、商品与订单上下文、演示预设问题；
- 主区：客户问题输入 -> 链路分步卡片（意图 / 闸门 / 事实 / 回复 / 校验 / 最终决策）
  -> 草稿采纳交互 -> 看板（使用统计 / 拦截记录 / 大促规则管理）。

运行：streamlit run src/serving/web_app.py
约定：远程推理服务连不上时显式报错，不静默降级；streamlit 运行时只走 main()，
模块本身可安全 import（单测不依赖 streamlit 运行时）。
"""
import json
from datetime import date, datetime
from pathlib import Path

import streamlit as st

from src.config import (
    DATA_PROCESSED,
    GENERATE_API_URL,
    INTENT_CONFIDENCE_THRESHOLD,
    get_llm_client,
)
from src.serving import fact_apis
from src.serving.generator import RemoteGenerator
from src.serving.intent import HIGH_RISK_INTENTS
from src.serving.pipeline import FACT_API_SOFT_REFUSAL, handle_query
from src.serving.promo_knowledge import PromoKnowledgeStore
from src.serving.stats import ADOPTION_ACTIONS, calc_adoption_rate, load_jsonl

# 采纳记录与决策日志（决策日志由 pipeline.handle_query 写入）
ADOPTION_LOG = DATA_PROCESSED / "adoption_log.jsonl"
DECISION_LOG = DATA_PROCESSED / "decision_log.jsonl"

# 演示预设问题：覆盖五个面试演示场景（一键填入输入框）
PRESET_QUESTIONS = [
    ("价格咨询自动发送", "这款多少钱？有优惠券吗"),
    ("V1 漏判 vs V2 正确", "现在买划算吗"),
    ("高危永远转人工", "我用了你们的面霜过敏了！烂脸了！"),
    ("物流查询", "我的快递到哪了"),
    ("大促规则（需先上传规则）", "双11定金和尾款怎么算，能叠加吗"),
]

# 大促规则示例：一键填入，便于演示「上传 -> 注入 -> 校验」流程
SAMPLE_PROMO_DOC = (
    "一、双11预售定金 50 元，支付定金可抵 100 元尾款。\n"
    "二、尾款支持跨店满 300 减 50，可与店铺券叠加使用。\n"
    "三、预售付定金时间：10月20日 20:00 至 10月31日 22:00，逾期未付定金不退。"
)


class MockGenerator:
    """本地模拟生成器：与推理服务相同的 generate(system, user, facts) 接口。

    用于本地演示（真实远程推理服务需用户在 AutoDL 部署后切换）：
    - 有 facts 时忠实引用事实组织品牌语气回复，回复中的数字全部来自 facts，
      保证数字校验通过（这是"事实与风格分离"的正面演示）；
    - 无 facts 时按关键词给产品咨询 / 安抚话术模板（不含数字，
      草稿通道由人工把关）。
    """

    def generate(self, system: str, user: str, facts: dict | None = None, **kwargs) -> str:
        if facts:
            if "price" in facts:  # 商品接口事实（价格咨询）
                coupon = facts["coupons"][0] if facts["coupons"] else None
                coupon_part = f"，进店还能领 {coupon}" if coupon else ""
                return (
                    f"宝子，这款现在 {facts['price']} 元哦（原价 {facts['original_price']}）"
                    f"{coupon_part}，今天下单超划算~"
                )
            if "status" in facts:  # 快递接口事实（物流查询）
                return (
                    f"宝子，您的快递现在{facts['status']}啦，"
                    f"{facts['latest']}，{facts['eta']}哦~"
                )
            if "活动规则" in facts:  # 大促规则条款（原样引用，保证条款事实校验通过）
                return (
                    f"宝子，这次活动规则是这样哒：{facts['活动规则']}"
                    "还有不清楚的随时问我哦~"
                )
        # 无事实注入：按关键词给产品咨询 / 安抚话术模板
        if any(k in user for k in ("肤质", "成分", "色号", "怎么用", "用法", "功效", "油皮", "干皮")):
            return (
                "宝子，咱家这款是温和配方哒，大多数肤质都能用哦~ "
                "您跟我说说您的肤质和想改善的问题，我帮您参谋参谋~"
            )
        if any(k in user for k in ("生气", "投诉", "差评", "无语", "气死", "烦")):
            return (
                "宝子消消气，让您不开心了真的很抱歉~ "
                "您跟我说说具体情况，我一定尽力帮您处理好哒~"
            )
        return "宝子您好呀，有什么可以帮您的吗？跟我讲讲您的需求，我来帮您看看~"


# ---------- 资源构造（缓存，跨 rerun 复用；缺失配置直接报错，不做降级） ----------

@st.cache_resource
def _get_promo_store() -> PromoKnowledgeStore:
    """大促规则存储（持久化在 data/processed/promo_docs.json）。"""
    return PromoKnowledgeStore()


@st.cache_resource
def _get_llm_client():
    """V2 意图分类的 LLM client；未配置 AGICTO_API_KEY 时抛错（界面展示，不降级）。"""
    return get_llm_client()


@st.cache_resource
def _get_remote_generator() -> RemoteGenerator:
    """远程推理服务客户端（AutoDL 部署的 adapter 服务）。"""
    return RemoteGenerator(GENERATE_API_URL)


# ---------- 侧边栏 ----------

def _render_sidebar() -> dict:
    """渲染侧边栏配置，返回链路运行所需参数。"""
    st.sidebar.header("链路配置")

    intent_label = st.sidebar.radio(
        "意图分类版本",
        ["v1_rules 关键词规则", "v2_llm LLM 分类"],
        index=1,
        help="演示迭代故事：V1 会被绕开关键词的问法漏判（如「现在买划算吗」），V2 用 LLM 前置分类修正",
    )
    intent_version = "v1" if intent_label.startswith("v1") else "v2"

    gen_label = st.sidebar.radio(
        "生成器模式",
        ["模拟生成器（本地演示）", "远程推理服务（AutoDL adapter）"],
        index=0,
        help="远程模式需先在 AutoDL 部署推理服务并配置 GENERATE_API_URL，连不上会显式报错",
    )
    generator_mode = "mock" if gen_label.startswith("模拟") else "remote"

    st.sidebar.header("上下文")
    product_options = ["（不选择）"] + [
        f"{pid} {info['name']}" for pid, info in fact_apis.PRODUCT_DB.items()
    ]
    product_choice = st.sidebar.selectbox("当前咨询商品", product_options)
    product_id = product_choice.split(" ")[0] if product_choice != "（不选择）" else None

    order_options = ["（不选择）"] + list(fact_apis.LOGISTICS_DB.keys())
    order_choice = st.sidebar.selectbox("当前咨询订单", order_options)
    order_id = order_choice if order_choice != "（不选择）" else None

    st.sidebar.header("演示预设问题")
    for label, question in PRESET_QUESTIONS:
        st.sidebar.button(
            label,
            on_click=_fill_query,
            args=(question,),
            width="stretch",
        )

    return {
        "intent_version": intent_version,
        "generator_mode": generator_mode,
        "product_id": product_id,
        "order_id": order_id,
    }


def _fill_query(question: str) -> None:
    """预设问题按钮回调：把问题填入输入框。"""
    st.session_state["query_input"] = question


# ---------- 链路执行 ----------

def _run_pipeline(query: str, cfg: dict):
    """执行在线链路，返回 PipelineResult；配置缺失 / 服务不可达时显式报错并返回 None。"""
    if cfg["generator_mode"] == "remote":
        generator = _get_remote_generator()
    else:
        generator = MockGenerator()

    client = None
    if cfg["intent_version"] == "v2":
        try:
            client = _get_llm_client()
        except RuntimeError as exc:
            st.error(f"V2 意图分类不可用：{exc}")
            return None

    try:
        return handle_query(
            query,
            product_id=cfg["product_id"],
            order_id=cfg["order_id"],
            intent_version=cfg["intent_version"],
            client=client,
            generator=generator,
            promo_store=_get_promo_store(),
        )
    except Exception as exc:
        # 远程推理服务连不上 / 生成失败：显式报错，不静默降级到模拟生成器
        st.error(f"链路执行失败：{type(exc).__name__}: {exc}")
        return None


def _gate_explanation(result) -> str:
    """步骤 2 闸门决策说明：双条件（高置信意图 + 该意图有事实接口）的逐项判定。"""
    ir = result.intent_result
    if ir.intent in HIGH_RISK_INTENTS:
        return "命中高危意图：无论置信度多高，永远转人工，不进入生成。"
    if ir.intent == "售后政策":
        return "售后政策为模板化答案意图：固定话术直接自动发送，不经生成与数字校验。"
    if ir.intent == "活动规则":
        return "活动规则意图：委托大促规则模块（检索条款 -> 注入生成 -> 条款事实校验）。"
    if ir.intent in FACT_API_SOFT_REFUSAL:
        if ir.confidence >= INTENT_CONFIDENCE_THRESHOLD:
            return (
                f"双条件通过：意图「{ir.intent}」有事实接口，"
                f"且置信度 {ir.confidence:.2f} ≥ 阈值 {INTENT_CONFIDENCE_THRESHOLD}，进入自动通道。"
            )
        return (
            f"双条件未通过：置信度 {ir.confidence:.2f} 低于阈值 {INTENT_CONFIDENCE_THRESHOLD}，"
            "走草稿通道由人工把关。"
        )
    return f"双条件未通过：意图「{ir.intent}」没有事实接口，走草稿通道由人工把关。"


_DECISION_BANNER = {
    "auto_send": ("success", "最终决策：自动发送（系统直接发给客户）"),
    "draft": ("warning", "最终决策：转草稿（等客服在下方确认后发出）"),
    "human": ("error", "最终决策：转人工（系统不生成，直接交给人）"),
}


def _render_result(query: str, result) -> None:
    """链路分步卡片展示。"""
    ir = result.intent_result

    with st.container(border=True):
        st.markdown("**步骤 1 ｜ 意图识别**")
        cols = st.columns(3)
        cols[0].metric("意图", ir.intent)
        cols[1].metric("置信度", f"{ir.confidence:.2f}")
        cols[2].metric("分类版本", ir.version)
        st.caption(f"判定依据：{ir.reason}")

    with st.container(border=True):
        st.markdown("**步骤 2 ｜ 闸门决策**")
        st.write(_gate_explanation(result))
        st.caption("规则：自动发送必须同时满足「高置信意图 + 该意图有事实接口」，其余留草稿；高危意图永远转人工。")

    with st.container(border=True):
        st.markdown("**步骤 3 ｜ 事实注入**")
        if result.facts:
            st.json(result.facts)
        elif ir.intent in FACT_API_SOFT_REFUSAL and result.decision != "auto_send":
            st.write("接口无数据，转人工 / 草稿确认。")
        else:
            st.write("本链路无事实注入（该意图无事实接口或走模板 / 规则模块）。")

    with st.container(border=True):
        st.markdown("**步骤 4 ｜ 模型回复**")
        st.write(result.reply if result.reply else "（无回复：直接转人工）")

    with st.container(border=True):
        st.markdown("**步骤 5 ｜ 数字校验**")
        fc = result.fact_check
        if not fc.get("checked"):
            st.write("本链路未触发数字校验（无生成或无接口事实）。")
        elif fc["passed"]:
            st.write("校验通过：回复中的数字全部能在接口返回值中找到。")
        else:
            st.write(f"校验拦截：回复含接口外的编造数字 {fc['unmatched']}，转草稿。")

    level, text = _DECISION_BANNER[result.decision]
    getattr(st, level)(f"**{text}**")

    with st.expander("完整决策依据（reason 链路串联）"):
        st.text(result.reason)

    if result.decision == "draft":
        _render_draft_actions(query, result)


def _record_adoption(query: str, decision: str, action: str) -> None:
    """写入采纳记录（jsonl），看板按此统计草稿采纳率。"""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "decision": decision,
        "action": action,
    }
    ADOPTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ADOPTION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _render_draft_actions(query: str, result) -> None:
    """草稿模式交互：直接发送 / 修改后发送 / 弃用。"""
    st.markdown("**草稿处理**（模拟客服在千牛侧的操作）")
    cols = st.columns(3)
    if cols[0].button("直接发送", width="stretch"):
        _record_adoption(query, result.decision, "直接发送")
        st.success("已发送（采纳记录已写入）")
    if cols[1].button("修改后发送", width="stretch"):
        st.session_state["editing_draft"] = True
    if cols[2].button("弃用", width="stretch"):
        _record_adoption(query, result.decision, "弃用")
        st.info("已弃用（采纳记录已写入）")

    if st.session_state.get("editing_draft"):
        st.text_area("修改草稿", value=result.reply, key="edited_reply")
        if st.button("确认发送修改稿"):
            _record_adoption(query, result.decision, "修改后发送")
            st.session_state["editing_draft"] = False
            st.success("修改稿已发送（采纳记录已写入）")


def _render_query_panel(cfg: dict) -> None:
    """主区：问题输入 + 发送 + 结果展示。"""
    st.text_area(
        "客户问题",
        key="query_input",
        height=80,
        placeholder="输入客户问题，或点左侧预设问题一键填入",
    )
    if st.button("发送", type="primary"):
        query = st.session_state.get("query_input", "").strip()
        if not query:
            st.warning("请先输入客户问题")
        else:
            result = _run_pipeline(query, cfg)
            if result is not None:
                st.session_state["last"] = {"query": query, "result": result}
                st.session_state.pop("editing_draft", None)

    last = st.session_state.get("last")
    if last:
        _render_result(last["query"], last["result"])


# ---------- 看板 ----------

def _render_dashboard() -> None:
    """看板区：使用统计 / 拦截记录 / 大促规则管理。"""
    tab_stats, tab_block, tab_promo = st.tabs(["使用统计", "拦截记录", "大促规则管理"])

    with tab_stats:
        stats = calc_adoption_rate(load_jsonl(ADOPTION_LOG))
        if stats["total"] == 0:
            st.info("暂无草稿采纳记录：先在上方触发一个转草稿的咨询，再点草稿处理按钮。")
        else:
            cols = st.columns(4)
            cols[0].metric("草稿总数", stats["total"])
            cols[1].metric("直接发送", stats["counts"]["直接发送"])
            cols[2].metric("修改后发送", stats["counts"]["修改后发送"])
            cols[3].metric("弃用", stats["counts"]["弃用"])
            st.metric("草稿采纳率", f"{stats['adoption_rate']:.0%}", help="（直接发送 + 修改后发送）/ 草稿总数")

    with tab_block:
        rows = [
            r for r in load_jsonl(DECISION_LOG)
            if r.get("decision") in ("draft", "human")
        ]
        if rows:
            st.caption("决策为草稿 / 转人工的记录（禁用语、编造数字、低置信等触发拦截），最新在前。")
            st.dataframe(rows[::-1], width="stretch")
        else:
            st.info("暂无拦截记录。")

    with tab_promo:
        _render_promo_panel()


def _fill_sample_promo() -> None:
    """填入示例双 11 规则文档。"""
    st.session_state["promo_content"] = SAMPLE_PROMO_DOC


def _render_promo_panel() -> None:
    """大促规则管理：上传规则文档 + 展示当前生效条款。"""
    store = _get_promo_store()
    st.caption("一年只用 2~4 次（双 11、618 等）：店长上传一页纸规则，按条款切块、标注有效期，过期自动下线。")

    st.text_input("规则文档标题", value="双11活动规则", key="promo_title")
    st.text_area(
        "规则文档内容（按「一、」「1.」等条款编号分段）",
        height=160,
        key="promo_content",
    )
    st.button("填入示例双11规则", on_click=_fill_sample_promo)
    cols = st.columns(2)
    start = cols[0].date_input("生效日期", value=date(2026, 10, 20))
    end = cols[1].date_input("下线日期", value=date(2026, 11, 15))
    if st.button("上传规则文档", type="primary"):
        content = st.session_state.get("promo_content", "").strip()
        if not content:
            st.warning("请先粘贴规则文档内容")
        else:
            clauses = store.upload_doc(
                st.session_state["promo_title"], content, start, end
            )
            st.success(f"已上传，切出 {len(clauses)} 条条款")

    st.markdown("**当前生效中的规则条款**")
    active = store.retrieve("", now=date.today())
    if active:
        for i, clause in enumerate(active, 1):
            st.write(f"{i}. {clause}")
        st.caption("条款总字数小于 1500 时全量注入；超过才按相似度取 top3。有效期过后检索层自动不可见（自动下线）。")
    else:
        st.info("当前没有生效中的规则条款：上传后立即可用；有效期过后会自动下线，提问活动规则将转人工。")


# ---------- 入口 ----------

def main() -> None:
    st.set_page_config(page_title="美妆客服 SFT 演示", layout="wide")
    st.title("美妆电商客服话术 SFT 微调 · 在线链路演示")
    st.caption(
        "定位：客服辅助工具，与千牛双窗口并排使用——"
        "高置信且有事实接口的咨询由系统自动发送，其余进草稿由人点发送；高危场景永远转人工。"
    )
    cfg = _render_sidebar()
    _render_query_panel(cfg)
    st.divider()
    _render_dashboard()


if __name__ == "__main__":
    main()
