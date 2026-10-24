# -*- coding: utf-8 -*-
"""数据管线总入口：依次跑 T3 -> T8 全流程。

用法：python -m src.data_pipeline.run_pipeline

说明：T6（LLM 判别）与 T7（LLM 打标）需要真实 LLM API，环境变量
LLM_API_KEY 未配置时 get_llm_client() 会直接报错提示，不做静默跳过。
（key 从环境变量 AGICTO_API_KEY 读取，见 src/config.py）
"""
from src.config import get_llm_client
from src.data_pipeline import (
    build_dataset,
    desensitize,
    filter_quality,
    generate_raw,
    label_category,
    split_sessions,
)


def main():
    print("=" * 60)
    print("[T3] 生成模拟原始数据")
    print("=" * 60)
    generate_raw.main()

    print("=" * 60)
    print("[T4] 脱敏")
    print("=" * 60)
    desensitize.main()

    print("=" * 60)
    print("[T5] 会话拆分")
    print("=" * 60)
    split_sessions.main()

    # T6/T7 依赖真实 LLM：key 缺失时在此明确报错（RuntimeError 提示设置 AGICTO_API_KEY）
    print("=" * 60)
    print("[T6/T7] 初始化 LLM 客户端（需环境变量 AGICTO_API_KEY）")
    print("=" * 60)
    client = get_llm_client()

    print("[T6] 优质会话筛选")
    filter_quality.main(client)

    print("=" * 60)
    print("[T7] 会话分类与配比")
    print("=" * 60)
    label_category.main(client)

    print("=" * 60)
    print("[T8] 占位符化与训练集构建")
    print("=" * 60)
    build_dataset.main()

    print("=" * 60)
    print("数据管线全流程完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
