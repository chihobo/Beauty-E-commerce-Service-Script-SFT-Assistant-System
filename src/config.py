# -*- coding: utf-8 -*-
"""项目集中配置模块。

约定：
- 敏感项（API key 等）一律从环境变量读取，缺失即报错，不做静默兜底；
- 路径常量基于项目根目录用 pathlib 构造，跨平台可用；
- 业务常量集中在此，后续数据管线 / 训练 / 在线服务 / 评估模块统一引用。
"""
import os
from pathlib import Path

from openai import OpenAI

# 项目根目录（本文件位于 src/config.py，上一级即仓库根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- LLM 配置（离线打标与在线意图分类共用，OpenAI 兼容接口） ----------
# 平台：AGICTO（https://agicto.com），统一 Base URL + Bearer Key 调用各家模型
# API key 必须显式设置，未设置时 get_llm_client() 直接报错
LLM_API_KEY = os.environ.get("AGICTO_API_KEY")
# OpenAI 兼容接口地址，默认 AGICTO 平台
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.agicto.cn/v1")
# 默认模型（AGICTO 模型广场中的模型名，可用环境变量覆盖）
# 已验证：deepseek-flash 适合本项目轻量高频打标/分类场景（快、直出 JSON、便宜）；
# deepseek-v4-pro 为深度思考模型，思考 token 开销大，不适合短输出分类任务
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-flash")

# ---------- 数据路径 ----------
DATA_RAW = PROJECT_ROOT / "data" / "raw"            # 模拟千牛原始数据
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # 管线各阶段产物
DATA_TRAIN = PROJECT_ROOT / "data" / "train"          # 训练集 / 测试集

# ---------- 业务常量 ----------
# 会话拆分：同一客户相邻消息超过该间隔即拆为新会话（分钟）
SESSION_GAP_MINUTES = 30
# 会话分类目标配比（售前降采样，售后/闲聊全保留）
CATEGORY_RATIO = {"售前": 0.6, "售后": 0.3, "闲聊安抚": 0.1}
# 在线意图分类置信度阈值：达到该值才允许进入双条件闸门判定
INTENT_CONFIDENCE_THRESHOLD = 0.8
# 训练集 / 测试集目标规模（demo 缩小版）
TRAIN_SIZE = 800
TEST_SIZE = 120
# 全局随机种子：数据生成 / 降采样 / 分层划分统一引用，保证可复现
SEED = 42

# ---------- 生成服务地址（推理服务，部署于 AutoDL 实例） ----------
GENERATE_API_URL = os.environ.get("GENERATE_API_URL", "http://127.0.0.1:8001")


def get_llm_client() -> OpenAI:
    """构造 OpenAI 兼容客户端。

    环境变量 AGICTO_API_KEY 未设置时直接抛出中文错误，
    项目约定不做兜底链路，配置缺失即视为环境问题。
    """
    if not LLM_API_KEY:
        raise RuntimeError(
            "未检测到环境变量 AGICTO_API_KEY，请先设置后再调用 LLM 相关功能"
        )
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
