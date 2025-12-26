"""
日志输出测试脚本

测试各个日志文件是否正确输出：
- app-*.log: 全量日志
- agent-*.log: Agent 智能推理日志
- rag-*.log: RAG 流水线日志
- ragas-eval-*.log: RAG 质量评估日志
- error-*.log: 错误日志
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置环境变量
os.environ.setdefault('LOG_LEVEL', 'DEBUG')

import logging
from app.core.logging_config import setup_logging


def test_logging_output():
    """测试各个日志文件输出"""

    print("=" * 60)
    print("日志输出测试")
    print("=" * 60)

    # 初始化日志系统
    setup_logging()

    today = datetime.now().strftime('%Y-%m-%d')
    log_dir = Path("logs")

    # 预期的日志文件
    expected_files = {
        f"app-{today}.log": "全量日志",
        f"agent-{today}.log": "Agent 智能推理日志",
        f"rag-{today}.log": "RAG 流水线日志",
        f"ragas-eval-{today}.log": "RAG 质量评估日志",
        f"error-{today}.log": "错误日志",
    }

    print(f"\n[1] 检查日志文件是否存在...")
    for filename, desc in expected_files.items():
        filepath = log_dir / filename
        exists = "✓" if filepath.exists() else "✗"
        print(f"  {exists} {filename} ({desc})")

    # 获取各个模块的 logger
    print(f"\n[2] 测试各模块日志输出...")

    # Agent 模块
    agent_logger = logging.getLogger('app.agent.langgraph_agent')
    agent_logger.info("[TEST] Agent 模块日志测试 - INFO 级别")
    agent_logger.debug("[TEST] Agent 模块日志测试 - DEBUG 级别")
    print("  → app.agent.langgraph_agent: 已写入")

    # RAG 模块
    rag_logger = logging.getLogger('app.rag.query_engine')
    rag_logger.info("[TEST] RAG 模块日志测试 - INFO 级别")
    rag_logger.debug("[TEST] RAG 模块日志测试 - DEBUG 级别")
    print("  → app.rag.query_engine: 已写入")

    # RAGAS 评估模块
    ragas_logger = logging.getLogger('app.rag.ragas_evaluator')
    ragas_logger.info("[TEST] RAGAS 评估模块日志测试 - INFO 级别")
    ragas_logger.debug("[TEST] RAGAS 评估模块日志测试 - DEBUG 级别")
    print("  → app.rag.ragas_evaluator: 已写入")

    # 错误日志
    error_logger = logging.getLogger('app.test')
    error_logger.error("[TEST] 错误日志测试 - ERROR 级别")
    print("  → app.test (error): 已写入")

    # 验证日志内容
    print(f"\n[3] 验证日志文件内容...")

    results = {}

    # 检查 agent 日志
    agent_log = log_dir / f"agent-{today}.log"
    if agent_log.exists():
        content = agent_log.read_text()
        has_test = "[TEST] Agent 模块日志测试" in content
        results["agent"] = has_test
        status = "✓" if has_test else "✗"
        print(f"  {status} agent-{today}.log: {'包含测试日志' if has_test else '未找到测试日志'}")
    else:
        results["agent"] = False
        print(f"  ✗ agent-{today}.log: 文件不存在")

    # 检查 rag 日志
    rag_log = log_dir / f"rag-{today}.log"
    if rag_log.exists():
        content = rag_log.read_text()
        has_test = "[TEST] RAG 模块日志测试" in content
        results["rag"] = has_test
        status = "✓" if has_test else "✗"
        print(f"  {status} rag-{today}.log: {'包含测试日志' if has_test else '未找到测试日志'}")
    else:
        results["rag"] = False
        print(f"  ✗ rag-{today}.log: 文件不存在")

    # 检查 ragas-eval 日志
    ragas_log = log_dir / f"ragas-eval-{today}.log"
    if ragas_log.exists():
        content = ragas_log.read_text()
        has_test = "[TEST] RAGAS 评估模块日志测试" in content
        results["ragas"] = has_test
        status = "✓" if has_test else "✗"
        print(f"  {status} ragas-eval-{today}.log: {'包含测试日志' if has_test else '未找到测试日志'}")
    else:
        results["ragas"] = False
        print(f"  ✗ ragas-eval-{today}.log: 文件不存在")

    # 检查 error 日志
    error_log = log_dir / f"error-{today}.log"
    if error_log.exists():
        content = error_log.read_text()
        has_test = "[TEST] 错误日志测试" in content
        results["error"] = has_test
        status = "✓" if has_test else "✗"
        print(f"  {status} error-{today}.log: {'包含测试日志' if has_test else '未找到测试日志'}")
    else:
        results["error"] = False
        print(f"  ✗ error-{today}.log: 文件不存在")

    # 检查 app 日志（应该包含所有日志）
    app_log = log_dir / f"app-{today}.log"
    if app_log.exists():
        content = app_log.read_text()
        has_all = all([
            "[TEST] Agent 模块日志测试" in content,
            "[TEST] RAG 模块日志测试" in content,
            "[TEST] RAGAS 评估模块日志测试" in content,
            "[TEST] 错误日志测试" in content,
        ])
        results["app"] = has_all
        status = "✓" if has_all else "✗"
        print(f"  {status} app-{today}.log: {'包含所有测试日志' if has_all else '缺少部分测试日志'}")
    else:
        results["app"] = False
        print(f"  ✗ app-{today}.log: 文件不存在")

    # 总结
    print(f"\n[4] 测试结果总结")
    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("✓ 所有日志输出测试通过!")
    else:
        print("✗ 部分日志输出测试失败:")
        for name, passed in results.items():
            if not passed:
                print(f"  - {name} 日志未正确输出")

    print("=" * 60)

    # 显示日志文件大小
    print(f"\n[5] 日志文件状态:")
    for filename in expected_files.keys():
        filepath = log_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  {filename}: {size:,} bytes")
        else:
            print(f"  {filename}: 不存在")

    return all_passed


if __name__ == "__main__":
    success = test_logging_output()
    sys.exit(0 if success else 1)