#!/usr/bin/env python
"""离线评测。作为发布门槛：分数不达标应当阻止上线。

评测是**批量离线**的事，不是请求路径上的服务。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ics.core.context import RequestContext

THRESHOLD = 0.7


async def collect(tenant: str, questions: list[dict], kb_id: str) -> list[dict]:
    from ics.api.wiring import build_container

    container, stack = await build_container()
    try:
        ctx = RequestContext(tenant_id=tenant)
        rows = []
        for item in questions:
            result = await container.retrieval.retrieve(ctx, item["question"], kb_id=kb_id)
            rows.append({
                "user_input": item["question"],
                "retrieved_contexts": [c.content for c in result.chunks],
                "reference": item.get("ground_truth", ""),
            })
        return rows
    finally:
        await stack.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--kb-id", default="default")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()

    questions = [
        json.loads(line)
        for line in args.dataset.read_text("utf-8").splitlines()
        if line.strip()
    ]
    rows = asyncio.run(collect(args.tenant, questions, args.kb_id))

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import ContextPrecision, ContextRecall
    except ImportError:
        print("未安装评测依赖：uv sync --extra eval", file=sys.stderr)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        raise SystemExit(2) from None

    result = evaluate(
        EvaluationDataset.from_list(rows), metrics=[ContextPrecision(), ContextRecall()]
    )
    print(result)
    scores = [v for v in dict(result).values() if isinstance(v, (int, float))]
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n综合得分 {mean:.3f}（门槛 {args.threshold}）")
    # 🔴 非 0 退出码 = 发布闸门。分数不达标不允许上线。
    raise SystemExit(0 if mean >= args.threshold else 1)


if __name__ == "__main__":
    main()
