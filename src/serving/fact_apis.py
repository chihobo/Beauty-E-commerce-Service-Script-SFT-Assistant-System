# -*- coding: utf-8 -*-
"""模拟事实接口：商品 API 与快递 API 的本地模拟实现。

说明：真实环境对接淘宝开放平台（商品价格/库存/券）与快递100（物流轨迹），
需要商家资质与授权，demo 拿不到，故用本地字典模拟。
返回结构与真实接口保持一致（字段名/取值形态对齐），链路上层不感知差异。
查不到时返回 None，供链路走拒答/转人工分支，不做兜底假数据。
"""

# 模拟商品库：价格用真实样式（现价/原价/库存/当前生效券）
PRODUCT_DB = {
    "P001": {
        "name": "玻尿酸保湿精华 30ml",
        "price": 129,
        "original_price": 169,
        "stock": 500,
        "coupons": ["满99减10"],
    },
    "P002": {
        "name": "烟酰胺提亮面霜 50g",
        "price": 169,
        "original_price": 219,
        "stock": 320,
        "coupons": ["满199减20", "满99减10"],
    },
    "P003": {
        "name": "氨基酸温和洁面乳 120g",
        "price": 69,
        "original_price": 89,
        "stock": 1200,
        "coupons": [],
    },
    "P004": {
        "name": "舒缓修护面膜 5 片装",
        "price": 99,
        "original_price": 129,
        "stock": 0,
        "coupons": ["满99减10"],
    },
}

# 模拟订单物流库：状态 / 最新节点 / 预计送达
LOGISTICS_DB = {
    "O20260618001": {
        "status": "运输中",
        "latest": "已到杭州转运中心",
        "eta": "明天送达",
    },
    "O20260618002": {
        "status": "已揽收",
        "latest": "快递已在上海青浦揽收",
        "eta": "预计 2~3 天送达",
    },
    "O20260618003": {
        "status": "已签收",
        "latest": "包裹已由本人签收",
        "eta": "已送达",
    },
}


def get_product_info(product_id: str) -> dict | None:
    """按商品 ID 查询现价/原价/库存/生效券；查不到返回 None。"""
    return PRODUCT_DB.get(product_id)


def get_logistics(order_id: str) -> dict | None:
    """按订单号查询物流状态/最新节点/预计送达；查不到返回 None。"""
    return LOGISTICS_DB.get(order_id)
