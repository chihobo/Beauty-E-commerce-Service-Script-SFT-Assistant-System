# -*- coding: utf-8 -*-
"""在线链路生成客户端（R9 / T13 客户端侧）：封装对推理服务的 HTTP 调用。

推理服务（src/training/infer_server.py）部署于 AutoDL 实例，serving 链路
通过本模块的 RemoteGenerator 请求生成。采用依赖注入设计：HTTP 会话对象
由构造时传入，单测可用桩对象替代，不依赖真实网络与服务。
"""
import requests

from src.config import GENERATE_API_URL


class RemoteGenerator:
    """推理服务客户端：POST /generate 获取品牌语气回复。

    base_url 默认取 config.GENERATE_API_URL（可用环境变量 GENERATE_API_URL
    覆盖）；session 为 requests.Session 风格的 HTTP 客户端，依赖注入便于测试。
    """

    def __init__(self, base_url: str = GENERATE_API_URL, session=None,
                 timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout

    def generate(self, system: str, user: str, facts: dict | None = None,
                 max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        """请求推理服务生成回复，返回 reply 文本。

        facts 为上游接口实时事实（如 {"现价": "129元"}），由服务端注入
        prompt，落实"事实与风格分离"；服务异常（连接失败/非 2xx）直接抛错，
        链路层自行决定转草稿或转人工，本层不做兜底。
        """
        payload = {
            "system": system,
            "user": user,
            "facts": facts,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        resp = self.session.post(
            f"{self.base_url}/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["reply"]
