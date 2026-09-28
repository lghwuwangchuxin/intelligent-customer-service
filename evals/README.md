# evals

**与 `src/` 平级，因为它是产品资产，不是测试代码。**

单元测试回答"代码有没有写错"，evals 回答"这个系统答得好不好"。
后者会随知识库、提示词、模型版本漂移，而且**失效时不报错**——只是答案慢慢变差。

## 三层稳定性里的「质量稳定」

| 层 | 手段 |
|---|---|
| 可用性稳定 | 健康探针、熔断、重试 |
| **质量稳定** | **本目录**：评测进 CI、发布门槛、召回率基线定时比对 |
| 成本稳定 | 预算熔断、缓存命中率监控、reasoning token 占比监控 |

## 用法

```bash
uv sync --extra eval
uv run python evals/run_eval.py --tenant demo --dataset evals/datasets/golden_v1.jsonl
```

退出码非 0 表示未达门槛，可直接用作发布闸门。
