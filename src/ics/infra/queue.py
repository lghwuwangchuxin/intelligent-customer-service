"""进程内队列。

🔴 队列必须有界。无界队列在"生产快、消费慢"时会把整个数据集读进内存 → OOM。
"""
from __future__ import annotations

import asyncio
from typing import Any

ingest_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=32)
trace_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)
