# -*- coding: utf-8 -*-
"""LLM 桩客户端：单元测试替代真实 OpenAI 客户端，不依赖外部 API。

用法：
    client = StubClient("yes")                      # 固定返回
    client = StubClient(lambda prompt: "售前")      # 按 prompt 动态返回
    client.chat.completions.calls                    # 已收到的 prompt 列表，可断言调用次数
"""


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responder):
        self._responder = responder
        self.calls = []

    def create(self, model, messages, **kwargs):
        prompt = "".join(m["content"] for m in messages)
        self.calls.append(prompt)
        content = self._responder(prompt) if callable(self._responder) else self._responder
        return _Response(content)


class _Chat:
    def __init__(self, responder):
        self.completions = _Completions(responder)


class StubClient:
    """与 OpenAI 客户端同构的桩：client.chat.completions.create(...)"""

    def __init__(self, responder="yes"):
        self.chat = _Chat(responder)
