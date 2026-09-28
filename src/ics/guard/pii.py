"""PII 检测规则。

🔴 编排框架内置的 PII 类型只有 ``email / credit_card / ip / mac_address / url``，
   **不含中国大陆手机号与身份证号**——而这两样恰恰是中文客服场景里最常见的 PII。
   照抄内置清单会让手机号原样进模型、进日志、进追踪。

这里只放**正则**（纯数据，不依赖任何框架），由 agent 层装配进中间件：
policy 归 guard，框架装配归 agent。
"""
from __future__ import annotations

# 中国大陆手机号：1 开头，第二位 3-9，共 11 位。用边界防止截取长数字串中间。
CN_MOBILE = r"(?<!\d)1[3-9]\d{9}(?!\d)"

# 中国大陆身份证：18 位，含出生日期校验位，末位可能是 X。
CN_ID_CARD = r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"

# 车牌（充电场景高频）：省份简称 + 字母 + 5~6 位，兼容新能源 6 位号
CN_PLATE = r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-HJ-NP-Z][A-HJ-NP-Z0-9]{5,6}"

# 银行卡：16~19 位连续数字
BANK_CARD = r"(?<!\d)\d{16,19}(?!\d)"

# 自定义类型 → 正则。框架内置的类型不在这里（用内置的即可）。
CUSTOM_DETECTORS: dict[str, str] = {
    "cn_mobile": CN_MOBILE,
    "cn_id_card": CN_ID_CARD,
    "cn_plate": CN_PLATE,
    "bank_card": BANK_CARD,
}

# 框架内置、可直接用名字引用的类型
BUILTIN_TYPES = frozenset({"email", "credit_card", "ip", "mac_address", "url"})


def detector_for(pii_type: str) -> str | None:
    """返回自定义正则；内置类型返回 None（让框架用自己的实现）。"""
    if pii_type in BUILTIN_TYPES:
        return None
    try:
        return CUSTOM_DETECTORS[pii_type]
    except KeyError:
        raise ValueError(
            f"未知 PII 类型 {pii_type!r}；内置: {sorted(BUILTIN_TYPES)}，"
            f"自定义: {sorted(CUSTOM_DETECTORS)}"
        ) from None
