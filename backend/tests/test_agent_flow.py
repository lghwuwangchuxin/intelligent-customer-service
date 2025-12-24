"""
Agent Flow Test / Agent 流程测试
================================

测试 Agent 推理流程的完整性和性能优化效果。

包含测试:
1. LLM Manager 超时和重试机制
2. Tool Registry 超时控制
3. Knowledge Search 缓存功能
4. 完整 Agent 推理流程
5. 性能基准测试

运行方式:
    python tests/test_agent_flow.py

Author: Intelligent Customer Service Team
Version: 1.0.0
"""
import asyncio
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_llm_timeout_retry():
    """测试 LLM Manager 超时和重试机制"""
    print("\n" + "=" * 60)
    print("测试 1: LLM Manager 超时和重试机制")
    print("=" * 60)

    from app.core.llm_manager import (
        LLMManager,
        DEFAULT_LLM_TIMEOUT,
        DEFAULT_MAX_RETRIES,
        DEFAULT_RETRY_DELAY,
    )

    # 验证默认配置
    assert DEFAULT_LLM_TIMEOUT == 120.0, f"默认超时应为 120s, 实际: {DEFAULT_LLM_TIMEOUT}"
    assert DEFAULT_MAX_RETRIES == 3, f"默认重试次数应为 3, 实际: {DEFAULT_MAX_RETRIES}"
    assert DEFAULT_RETRY_DELAY == 1.0, f"默认重试延迟应为 1s, 实际: {DEFAULT_RETRY_DELAY}"
    print(f"✓ 默认配置验证通过")
    print(f"  - 超时: {DEFAULT_LLM_TIMEOUT}s")
    print(f"  - 最大重试: {DEFAULT_MAX_RETRIES} 次")
    print(f"  - 重试延迟: {DEFAULT_RETRY_DELAY}s (指数退避)")

    # 验证 ainvoke 方法签名
    import inspect
    sig = inspect.signature(LLMManager.ainvoke)
    params = list(sig.parameters.keys())
    assert 'timeout' in params, "ainvoke 应有 timeout 参数"
    assert 'max_retries' in params, "ainvoke 应有 max_retries 参数"
    print(f"✓ ainvoke 方法签名验证通过")
    print(f"  - 参数: {params}")

    return True


async def test_llm_invoke_with_timeout():
    """测试 LLM 调用（需要 Ollama 运行）"""
    print("\n" + "=" * 60)
    print("测试 2: LLM 实际调用（带超时）")
    print("=" * 60)

    try:
        from app.core.llm_manager import LLMManager

        manager = LLMManager(
            provider="ollama",
            model="qwen3:latest",
            temperature=0.1,
            max_tokens=50,
        )
        print(f"✓ LLM Manager 初始化成功")
        print(f"  - 提供商: {manager.provider}")
        print(f"  - 模型: {manager.model}")

        # 测试带超时的调用
        messages = [{"role": "user", "content": "你好，请简短回复"}]

        start = time.time()
        response = await manager.ainvoke(
            messages,
            timeout=30.0,  # 30 秒超时
            max_retries=1,  # 1 次重试
        )
        elapsed = time.time() - start

        assert response is not None, "响应不应为空"
        assert len(response) > 0, "响应长度应大于 0"
        print(f"✓ LLM 调用成功")
        print(f"  - 响应长度: {len(response)} 字符")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 响应预览: {response[:50]}...")

        return True

    except Exception as e:
        print(f"⚠ LLM 调用测试跳过: {e}")
        print("  (可能 Ollama 未运行或模型未加载)")
        return False


def test_tool_registry_timeout():
    """测试 Tool Registry 超时控制"""
    print("\n" + "=" * 60)
    print("测试 3: Tool Registry 超时控制")
    print("=" * 60)

    from app.mcp.registry import ToolRegistry, DEFAULT_TOOL_TIMEOUT

    # 验证默认配置
    assert DEFAULT_TOOL_TIMEOUT == 60.0, f"默认超时应为 60s, 实际: {DEFAULT_TOOL_TIMEOUT}"
    print(f"✓ 默认超时配置: {DEFAULT_TOOL_TIMEOUT}s")

    # 验证 execute 方法签名
    import inspect
    sig = inspect.signature(ToolRegistry.execute)
    params = list(sig.parameters.keys())
    assert 'timeout' in params, "execute 应有 timeout 参数"
    print(f"✓ execute 方法签名验证通过")
    print(f"  - 参数: {params}")

    # 创建测试工具
    registry = ToolRegistry()
    print(f"✓ ToolRegistry 创建成功")

    return True


async def test_tool_registry_execute():
    """测试 Tool Registry 执行（异步）"""
    from app.mcp.registry import ToolRegistry

    registry = ToolRegistry()

    # 测试不存在的工具
    result = await registry.execute("nonexistent_tool")
    assert result["success"] is False
    assert "not found" in result["error"]
    print(f"✓ 不存在工具处理正确")
    return True


def test_knowledge_cache():
    """测试 Knowledge Search 缓存功能"""
    print("\n" + "=" * 60)
    print("测试 4: Knowledge Search 缓存功能")
    print("=" * 60)

    from app.mcp.tools.knowledge import (
        SearchResultCache,
        CACHE_MAX_SIZE,
        CACHE_TTL,
        get_search_cache,
    )

    # 验证默认配置
    assert CACHE_MAX_SIZE == 100, f"最大缓存应为 100, 实际: {CACHE_MAX_SIZE}"
    assert CACHE_TTL == 300, f"TTL 应为 300s, 实际: {CACHE_TTL}"
    print(f"✓ 缓存配置验证通过")
    print(f"  - 最大条目: {CACHE_MAX_SIZE}")
    print(f"  - TTL: {CACHE_TTL}s")

    # 测试缓存基本功能
    cache = SearchResultCache(max_size=10, ttl=5)

    # 测试设置和获取
    test_query = "如何申请退款"
    test_results = [{"content": "退款流程...", "score": 0.9}]

    cache.set(test_query, 5, test_results)
    retrieved = cache.get(test_query, 5)

    assert retrieved is not None, "缓存应该命中"
    assert len(retrieved) == 1, "应返回 1 个结果"
    assert retrieved[0]["content"] == "退款流程...", "内容应匹配"
    print(f"✓ 缓存基本功能正常")

    # 测试缓存未命中
    miss_result = cache.get("不存在的查询", 5)
    assert miss_result is None, "应该缓存未命中"
    print(f"✓ 缓存未命中处理正确")

    # 测试缓存统计
    stats = cache.get_stats()
    assert stats["hits"] == 1, f"命中数应为 1, 实际: {stats['hits']}"
    assert stats["misses"] == 1, f"未命中数应为 1, 实际: {stats['misses']}"
    assert stats["hit_rate"] == 0.5, f"命中率应为 0.5, 实际: {stats['hit_rate']}"
    print(f"✓ 缓存统计正常")
    print(f"  - 命中: {stats['hits']}")
    print(f"  - 未命中: {stats['misses']}")
    print(f"  - 命中率: {stats['hit_rate']}")

    # 测试 LRU 驱逐
    for i in range(15):
        cache.set(f"query_{i}", 5, [{"content": f"result_{i}"}])

    assert len(cache._cache) <= 10, "缓存应不超过 max_size"
    print(f"✓ LRU 驱逐正常 (当前缓存: {len(cache._cache)} 条)")

    # 测试全局缓存实例
    global_cache = get_search_cache()
    assert global_cache is not None, "全局缓存应存在"
    print(f"✓ 全局缓存实例正常")

    return True


async def test_knowledge_search_with_cache():
    """测试 Knowledge Search 实际搜索（带缓存）"""
    print("\n" + "=" * 60)
    print("测试 5: Knowledge Search 实际搜索（带缓存）")
    print("=" * 60)

    from app.mcp.tools.knowledge import KnowledgeSearchTool, get_search_cache

    # 清空缓存
    cache = get_search_cache()
    cache.clear()

    # 创建搜索工具（无 RAG 服务，只测试文本搜索）
    tool = KnowledgeSearchTool()
    print(f"✓ KnowledgeSearchTool 初始化成功")

    query = "客服工作时间"

    # 第一次搜索（应该未命中缓存）
    start1 = time.time()
    results1 = await tool.execute(query, top_k=3, use_cache=True)
    time1 = time.time() - start1
    print(f"✓ 第一次搜索完成: {len(results1)} 个结果, 耗时: {time1*1000:.0f}ms")

    # 第二次搜索（应该命中缓存）
    start2 = time.time()
    results2 = await tool.execute(query, top_k=3, use_cache=True)
    time2 = time.time() - start2
    print(f"✓ 第二次搜索完成: {len(results2)} 个结果, 耗时: {time2*1000:.0f}ms")

    # 验证缓存效果
    if time2 < time1:
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"✓ 缓存加速: {speedup:.1f}x")
    else:
        print(f"⚠ 缓存未显著加速 (可能首次搜索很快)")

    # 检查缓存状态
    stats = cache.get_stats()
    print(f"✓ 缓存状态: {stats}")

    return True


async def test_agent_react_loop():
    """测试 Agent ReAct 推理循环"""
    print("\n" + "=" * 60)
    print("测试 6: Agent ReAct 推理循环")
    print("=" * 60)

    try:
        from app.agent.react_agent import ReActAgent
        from app.core.llm_manager import LLMManager
        from app.mcp.registry import ToolRegistry

        # 创建 mock LLM
        mock_llm = MagicMock(spec=LLMManager)
        mock_llm.model = "mock-model"
        mock_llm.ainvoke = AsyncMock(return_value="思考: 用户询问退款问题\n行动: 回答")

        # 创建工具注册表
        registry = ToolRegistry()

        # 创建 Agent
        agent = ReActAgent(
            llm_manager=mock_llm,
            tool_registry=registry,
            max_iterations=3,
        )
        print(f"✓ ReActAgent 初始化成功")
        print(f"  - 最大迭代: {agent.max_iterations}")
        print(f"  - 工具数量: {len(registry.get_all())}")

        # 运行简单查询
        result = await agent.run(
            question="如何申请退款？",
            conversation_id="test_session_001",
        )

        assert result is not None, "结果不应为空"
        assert "response" in result, "应包含 response"
        assert "iterations" in result, "应包含 iterations"
        print(f"✓ Agent 运行成功")
        print(f"  - 迭代次数: {result.get('iterations', 0)}")
        print(f"  - 工具调用: {len(result.get('tool_calls', []))}")
        print(f"  - 响应长度: {len(result.get('response', ''))}")

        return True

    except Exception as e:
        print(f"✗ Agent 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_agent_with_tools():
    """测试完整 Agent 流程（带工具）"""
    print("\n" + "=" * 60)
    print("测试 7: 完整 Agent 流程（带工具）")
    print("=" * 60)

    try:
        from app.agent.react_agent import ReActAgent
        from app.core.llm_manager import get_llm_manager
        from app.mcp.registry import get_tool_registry

        # 获取 LLM Manager
        llm = get_llm_manager()
        print(f"✓ LLM Manager: {llm.provider}/{llm.model}")

        # 获取工具注册表
        registry = get_tool_registry()
        # 初始化默认工具（不传入 RAG service，只有基本工具）
        print(f"✓ Tool Registry: {len(registry.get_all())} 个工具")

        # 创建 Agent
        agent = ReActAgent(
            llm_manager=llm,
            tool_registry=registry,
            max_iterations=5,
        )
        print(f"✓ ReActAgent 创建成功")

        # 测试简单对话
        question = "你好，请介绍一下你自己"
        print(f"\n执行查询: {question}")

        start = time.time()
        result = await agent.run(
            question=question,
            conversation_id="test_full_flow_001",
        )
        elapsed = time.time() - start

        print(f"\n✓ Agent 执行完成")
        print(f"  - 耗时: {elapsed:.2f}s")
        print(f"  - 迭代: {result.get('iterations', 0)}")
        print(f"  - 工具调用: {len(result.get('tool_calls', []))}")
        if result.get('response'):
            print(f"  - 响应预览: {result['response'][:100]}...")

        return True

    except Exception as e:
        print(f"⚠ 完整 Agent 测试失败: {e}")
        print("  (可能 LLM 服务未运行)")
        return False


def test_performance_benchmarks():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("测试 8: 性能基准测试")
    print("=" * 60)

    from app.mcp.tools.knowledge import SearchResultCache

    # 缓存性能测试
    cache = SearchResultCache(max_size=1000, ttl=300)
    test_data = [{"content": f"test content {i}", "score": 0.9} for i in range(10)]

    # 写入性能
    write_start = time.time()
    for i in range(1000):
        cache.set(f"query_{i}", 5, test_data)
    write_time = (time.time() - write_start) * 1000
    print(f"✓ 缓存写入: 1000 次, {write_time:.2f}ms ({write_time/1000:.3f}ms/次)")

    # 读取性能
    read_start = time.time()
    for i in range(1000):
        cache.get(f"query_{i}", 5)
    read_time = (time.time() - read_start) * 1000
    print(f"✓ 缓存读取: 1000 次, {read_time:.2f}ms ({read_time/1000:.3f}ms/次)")

    # 缓存命中率
    stats = cache.get_stats()
    print(f"✓ 缓存统计: 命中率 {stats['hit_rate']*100:.1f}%")

    return True


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Agent 流程优化验证测试")
    print("=" * 60)
    print(f"LLM 提供商: {settings.LLM_PROVIDER}")
    print(f"LLM 模型: {settings.LLM_MODEL}")

    results = {}

    # 测试 1: LLM 超时重试配置
    try:
        results['llm_config'] = test_llm_timeout_retry()
    except Exception as e:
        print(f"✗ LLM 配置测试失败: {e}")
        results['llm_config'] = False

    # 测试 2: LLM 实际调用
    try:
        results['llm_invoke'] = await test_llm_invoke_with_timeout()
    except Exception as e:
        print(f"✗ LLM 调用测试失败: {e}")
        results['llm_invoke'] = False

    # 测试 3: Tool Registry 超时
    try:
        results['tool_timeout'] = test_tool_registry_timeout()
    except Exception as e:
        print(f"✗ Tool 超时测试失败: {e}")
        results['tool_timeout'] = False

    # 测试 3b: Tool Registry 执行（异步）
    try:
        results['tool_execute'] = await test_tool_registry_execute()
    except Exception as e:
        print(f"✗ Tool 执行测试失败: {e}")
        results['tool_execute'] = False

    # 测试 4: Knowledge Cache
    try:
        results['knowledge_cache'] = test_knowledge_cache()
    except Exception as e:
        print(f"✗ Knowledge 缓存测试失败: {e}")
        results['knowledge_cache'] = False

    # 测试 5: Knowledge Search
    try:
        results['knowledge_search'] = await test_knowledge_search_with_cache()
    except Exception as e:
        print(f"✗ Knowledge 搜索测试失败: {e}")
        results['knowledge_search'] = False

    # 测试 6: Agent ReAct
    try:
        results['agent_react'] = await test_agent_react_loop()
    except Exception as e:
        print(f"✗ Agent ReAct 测试失败: {e}")
        results['agent_react'] = False

    # 测试 7: 完整 Agent 流程
    try:
        results['agent_full'] = await test_full_agent_with_tools()
    except Exception as e:
        print(f"✗ 完整 Agent 测试失败: {e}")
        results['agent_full'] = False

    # 测试 8: 性能基准
    try:
        results['performance'] = test_performance_benchmarks()
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        results['performance'] = False

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"通过: {passed}/{total}")

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
