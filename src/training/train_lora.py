# -*- coding: utf-8 -*-
"""LoRA 训练脚本（R8 / T12）：Qwen2.5-7B-Instruct + peft + trl SFTTrainer。

【AutoDL 环境准备步骤】（用户在 AutoDL 实例上执行，本地开发机不跑训练）
1. 选单卡 RTX 4090（24GB）实例，镜像选 PyTorch 2.x（CUDA 12.x）；
2. 安装训练依赖：pip install -r requirements-train.txt；
3. 模型下载（二选一）：
   - 实例可访问 HuggingFace：直接用默认 --model_name Qwen/Qwen2.5-7B-Instruct；
   - 无法直连：开启 AutoDL 学术资源加速，或用 modelscope 预下载：
       pip install modelscope
       modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /root/autodl-tmp/qwen2.5-7b
     然后以 --model_name /root/autodl-tmp/qwen2.5-7b 启动；
4. 运行训练：
       python src/training/train_lora.py --data_path data/train/train.jsonl

【本地开发约定】
本地无 GPU、不安装 torch。本文件重依赖（torch / transformers / peft / trl /
datasets）全部在 main() 内部 import，模块级只保留纯逻辑函数
（数据读取、messages 校验、chat template 格式化、参数解析器），
保证本地可静态 import 并进行单元测试。
"""
import argparse
import json
from pathlib import Path

# 训练数据默认路径（相对项目根）
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = "autodl-tmp/adapter"

# LoRA 目标模块：覆盖注意力与 MLP 的全部线性层。
# 理由：本任务只学品牌语气与话术风格、不改模型知识，低秩注入足够；
# 全 target 线性层相比只调 q/v_proj 表达容量更高，800 条小数据上
# 收敛更稳，rank=16 下参数量与显存开销仍然很小（面试问答有完整对比）。
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（纯逻辑，便于单测默认值）。"""
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct LoRA SFT 训练")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME,
                        help="基座模型名或本地路径")
    parser.add_argument("--data_path", required=True,
                        help="训练集 jsonl 路径（每行 {\"messages\": [...]}）")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="adapter 输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="单卡 per_device batch size")
    return parser


def load_jsonl(data_path: Path) -> list:
    """读取 jsonl 文件，返回样本列表（每行一个 JSON 对象）。"""
    data_path = Path(data_path)
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return samples


def validate_messages(messages, index: int = 0) -> None:
    """校验单条样本的 messages 结构，不合法直接抛错（不做兜底修复）。

    约定格式：首条为 system（品牌人设），其后 user/assistant 交替多轮，
    末条必须为 assistant（客服话术，即训练目标）。
    """
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"样本 {index}：messages 至少需要 system + 一轮 user/assistant")
    if messages[0].get("role") != "system":
        raise ValueError(f"样本 {index}：首条消息必须是 system（品牌人设）")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"样本 {index}：末条消息必须是 assistant（客服话术）")
    for pos, msg in enumerate(messages):
        if msg.get("role") not in ("system", "user", "assistant"):
            raise ValueError(f"样本 {index} 第 {pos} 条：非法 role={msg.get('role')!r}")
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            raise ValueError(f"样本 {index} 第 {pos} 条：content 不能为空")


def format_sample(messages, tokenizer) -> str:
    """用 tokenizer 的 chat template 把 messages 转成训练文本。"""
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_and_format_data(data_path, tokenizer) -> list:
    """读 jsonl → 校验 → chat template 格式化，返回 [{"text": ...}, ...]。

    tokenizer 只需提供 apply_chat_template 方法，单测可用桩对象替代。
    """
    samples = load_jsonl(Path(data_path))
    formatted = []
    for idx, sample in enumerate(samples):
        messages = sample.get("messages")
        validate_messages(messages, index=idx)
        formatted.append({"text": format_sample(messages, tokenizer)})
    return formatted


def main() -> None:
    """训练入口：重依赖全部在此 import，本地无 GPU 环境不影响 import 本模块。"""
    args = build_arg_parser().parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 基座模型：bf16 加载，4090 24GB 下单卡可训 LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # 训练时关闭 KV cache，配合 gradient_checkpointing 省显存

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,   # alpha = 2 * rank，缩放系数 alpha/r = 2，常用经验配比
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 训练超参（800 条 demo 数据、等效 batch 16 下约 150 步，小时级收敛）：
    # batch 4 * 梯度累积 4 = 等效 batch 16；cosine 退火；按 epoch 保存 adapter。
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        report_to=[],  # 不接外部看板，日志用 Trainer 自带 log_history 落盘
    )

    dataset = Dataset.from_list(load_and_format_data(args.data_path, tokenizer))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()

    # 只保存 adapter 权重（基座不重复落盘，adapter 通常百 MB 级）
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 训练日志落盘，供后续画 loss 曲线（评估/演示用）
    log_path = output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"训练完成，adapter 已保存到 {output_dir}，日志见 {log_path}")


if __name__ == "__main__":
    main()
