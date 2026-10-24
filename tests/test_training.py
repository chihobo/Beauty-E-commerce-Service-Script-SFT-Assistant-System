# -*- coding: utf-8 -*-
"""训练与推理相关用例（T12 / T13 的本地可验证部分）。

本地无 GPU、不装 torch，真跑训练与在线生成在 AutoDL 实例执行；
本文件覆盖纯逻辑：数据加载与格式化、build_prompt 事实注入、
RemoteGenerator 请求结构与返回解析、命令行参数默认值。
"""
import json

import pytest

from src.serving.generator import RemoteGenerator
from src.training import infer_server, train_lora


class _StubTokenizer:
    """假 tokenizer：模拟 apply_chat_template 的拼接行为。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _valid_sample():
    return {
        "messages": [
            {"role": "system", "content": "你是某美妆品牌的客服小助手。"},
            {"role": "user", "content": "这款精华现在多少钱？"},
            {"role": "assistant", "content": "亲亲，这款精华现价[当前活动价]哦。"},
        ]
    }


# ---------- T12：训练数据加载与格式化 ----------

def test_load_and_format_data_ok(tmp_path):
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, [_valid_sample(), _valid_sample()])

    formatted = train_lora.load_and_format_data(data_path, _StubTokenizer())

    assert len(formatted) == 2
    text = formatted[0]["text"]
    assert "<system>你是某美妆品牌的客服小助手。</system>" in text
    assert "<assistant>亲亲，这款精华现价[当前活动价]哦。</assistant>" in text


def test_validate_messages_missing_system(tmp_path):
    sample = _valid_sample()
    # 把首条换成 user（保持 3 条长度，单独触发 system 校验）
    sample["messages"][0] = {"role": "user", "content": "在吗？"}
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, [sample])

    with pytest.raises(ValueError, match="首条消息必须是 system"):
        train_lora.load_and_format_data(data_path, _StubTokenizer())


def test_validate_messages_last_not_assistant(tmp_path):
    sample = _valid_sample()
    sample["messages"].append({"role": "user", "content": "在吗？"})
    data_path = tmp_path / "train.jsonl"
    _write_jsonl(data_path, [sample])

    with pytest.raises(ValueError, match="末条消息必须是 assistant"):
        train_lora.load_and_format_data(data_path, _StubTokenizer())


def test_load_jsonl_bad_json(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="不是合法 JSON"):
        train_lora.load_jsonl(data_path)


# ---------- T12：训练脚本命令行参数默认值 ----------

def test_train_argparse_defaults():
    args = train_lora.build_arg_parser().parse_args(["--data_path", "x.jsonl"])
    assert args.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert args.output_dir == "autodl-tmp/adapter"
    assert args.epochs == 3
    assert args.batch_size == 4


def test_infer_argparse_defaults():
    args = infer_server.build_arg_parser().parse_args([])
    assert args.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.adapter_path is None
    assert args.port == 8001


# ---------- T13：build_prompt 事实注入 ----------

def test_build_prompt_with_facts():
    facts = {"现价": "129元", "优惠券": "满99减10"}
    prompt = infer_server.build_prompt("现在买有优惠吗？", facts)

    assert "请用品牌语气告知客户以下信息" in prompt
    assert "- 现价：129元" in prompt
    assert "- 优惠券：满99减10" in prompt
    # 防编造约束文案必须存在
    assert "禁止编造其他数字" in prompt
    assert prompt.endswith("客户问题：现在买有优惠吗？")


def test_build_prompt_without_facts():
    # 无 facts（如闲聊安抚类意图）时退化为纯客户问题
    assert infer_server.build_prompt("你们家好用吗？") == "你们家好用吗？"
    assert infer_server.build_prompt("你们家好用吗？", {}) == "你们家好用吗？"


# ---------- T13：RemoteGenerator 客户端 ----------

class _StubResponse:
    def __init__(self, payload):
        self._payload = payload
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    def json(self):
        return self._payload


class _StubSession:
    """假 HTTP 会话：记录请求并返回固定响应。"""

    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _StubResponse({"reply": "亲亲，现价129元哦。"})


def test_remote_generator_request_and_parse():
    session = _StubSession()
    gen = RemoteGenerator(base_url="http://127.0.0.1:8001/", session=session)

    reply = gen.generate(
        system="品牌人设",
        user="现在多少钱？",
        facts={"现价": "129元"},
    )

    assert reply == "亲亲，现价129元哦。"
    assert len(session.calls) == 1
    call = session.calls[0]
    # base_url 尾部斜杠被规整，请求路径正确
    assert call["url"] == "http://127.0.0.1:8001/generate"
    assert call["json"]["system"] == "品牌人设"
    assert call["json"]["user"] == "现在多少钱？"
    assert call["json"]["facts"] == {"现价": "129元"}
    assert call["json"]["max_new_tokens"] == 256
    assert call["json"]["temperature"] == 0.7
    assert call["timeout"] == 60
