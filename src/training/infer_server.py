# -*- coding: utf-8 -*-
"""推理服务（R9 / T13）：FastAPI 部署 base + LoRA adapter，部署于 AutoDL 实例。

【启动方式】（AutoDL 实例上执行）
    python src/training/infer_server.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path autodl-tmp/adapter \
        --port 8001

【设计要点：事实与风格分离】
训练语料中的价格/库存/优惠等易变事实已全部占位符化（见 R6），模型只学品牌
语气与话术风格；在线推理时由上游链路调用商品/快递接口拿到实时事实，通过
本服务的 facts 字段注入 prompt（见 build_prompt），并要求模型不得编造
接口之外的数字。输出侧另有数字校验模块兜底（R12），从机制上杜绝报错价。

【本地开发约定】
torch / transformers / peft 等重依赖全部在 lifespan 内 import，本地无 GPU
环境可静态 import 本模块，build_prompt 等纯逻辑可单元测试；模型加载失败
直接抛错终止启动，不做兜底。
"""
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# lifespan 加载完成后填充的运行时状态（tokenizer / model / 基座名）
_MODEL_STATE: dict = {}
# 启动参数（main() 解析后写入，lifespan 读取）
_SERVER_CONFIG: dict = {"base_model": None, "adapter_path": None}


def build_prompt(user: str, facts: dict | None = None) -> str:
    """组装发给模型的 user 侧内容（事实与风格分离的在线落地关键）。

    facts 为上游接口返回的实时事实（如 {"现价": "129元", "优惠券": "满99减10"}），
    格式化为注入块放在客户问题之前，并显式约束模型只能使用注入的数字；
    无 facts 时退化为纯客户问题（如闲聊安抚类意图不需要事实）。
    """
    if not facts:
        return user
    fact_lines = "\n".join(f"- {key}：{value}" for key, value in facts.items())
    return (
        "请用品牌语气告知客户以下信息：\n"
        f"{fact_lines}\n"
        "回复中的数字必须严格来自以上信息，禁止编造其他数字。\n\n"
        f"客户问题：{user}"
    )


class GenerateRequest(BaseModel):
    """POST /generate 入参。"""

    system: str                      # 品牌人设 system prompt
    user: str                        # 客户问题
    facts: dict | None = None        # 上游接口注入的实时事实（可选）
    max_new_tokens: int = 256
    temperature: float = 0.7


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动时加载基座与 adapter（bf16、device_map=auto），失败直接抛错。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = _SERVER_CONFIG["base_model"]
    adapter_path = _SERVER_CONFIG["adapter_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        # 加载 LoRA adapter；不传 adapter_path 时为纯基座（用于 SFT 前后对比评估）
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    _MODEL_STATE["tokenizer"] = tokenizer
    _MODEL_STATE["model"] = model
    _MODEL_STATE["base_model"] = base_model
    yield
    _MODEL_STATE.clear()


app = FastAPI(title="美妆客服 SFT 推理服务", lifespan=_lifespan)


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """注入事实上下文生成品牌语气回复。"""
    import torch

    tokenizer = _MODEL_STATE["tokenizer"]
    model = _MODEL_STATE["model"]

    messages = [
        {"role": "system", "content": req.system},
        {"role": "user", "content": build_prompt(req.user, req.facts)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只解码新生成部分，去掉 prompt 回显
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {"reply": reply}


@app.get("/health")
def health() -> dict:
    """健康检查：报告服务状态与已加载的基座模型。"""
    return {"status": "ok", "model": _MODEL_STATE.get("base_model")}


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（纯逻辑，便于单测）。"""
    parser = argparse.ArgumentParser(description="美妆客服 SFT 推理服务")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="基座模型名或本地路径")
    parser.add_argument("--adapter_path", default=None,
                        help="LoRA adapter 目录；不传则加载纯基座（SFT 前后对比用）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _SERVER_CONFIG["base_model"] = args.base_model
    _SERVER_CONFIG["adapter_path"] = args.adapter_path

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
