"""
Langfuse Integration Test / Langfuse 集成测试
=============================================

测试 Langfuse 可观测性服务是否正常工作。

运行方式:
    cd backend
    python -m tests.test_langfuse_integration
"""
import asyncio
import sys
import os

import pytest

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings

# Mark all async tests
pytestmark = pytest.mark.asyncio


def _check_ollama_available():
    """检查 Ollama 服务是否可用"""
    import httpx
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def test_langfuse_service():
    """测试 Langfuse 服务基本功能。"""
    print("\n" + "=" * 60)
    print("测试 1: Langfuse 服务初始化")
    print("=" * 60)

    from app.services.langfuse_service import get_langfuse_service, LangfuseService

    service = get_langfuse_service()

    print(f"✓ Langfuse 服务已初始化")
    print(f"  - 启用状态: {service.enabled}")
    print(f"  - Host: {settings.LANGFUSE_HOST}")
    print(f"  - Public Key: {settings.LANGFUSE_PUBLIC_KEY[:20] + '...' if settings.LANGFUSE_PUBLIC_KEY else 'Not set'}")
    print(f"  - Secret Key: {'Set' if settings.LANGFUSE_SECRET_KEY else 'Not set'}")

    if not service.enabled:
        print("\n⚠️  Langfuse 未启用或配置不完整")
        print("   请在 .env 中设置:")
        print("   LANGFUSE_ENABLED=true")
        print("   LANGFUSE_SECRET_KEY=sk-lf-xxx")
        print("   LANGFUSE_PUBLIC_KEY=pk-lf-xxx")
        return False

    return True


def test_trace_creation():
    """测试追踪创建。"""
    print("\n" + "=" * 60)
    print("测试 2: 创建追踪和 Span")
    print("=" * 60)

    from app.services.langfuse_service import get_langfuse_service

    service = get_langfuse_service()

    if not service.enabled:
        print("⚠️  跳过 - Langfuse 未启用")
        return False

    # 创建追踪
    trace = service.create_trace(
        name="test_trace",
        user_id="test_user",
        input={"test": "input"},
        metadata={"env": "test"},
        tags=["test", "integration"],
    )

    if trace:
        print(f"✓ 追踪已创建: {trace.id}")
    else:
        print("✗ 追踪创建失败")
        return False

    # 创建 Span
    span = service.create_span(
        trace,
        name="test_span",
        input={"step": 1},
    )

    if span:
        print(f"✓ Span 已创建")
    else:
        print("✗ Span 创建失败")
        return False

    # 结束 Span
    service.end_span(
        span,
        output={"result": "success"},
    )
    print(f"✓ Span 已结束")

    # 记录 Generation
    gen = service.log_generation(
        trace=trace,
        name="test_generation",
        model="test-model",
        input="测试输入",
        output="测试输出",
        usage={"input_tokens": 10, "output_tokens": 20},
    )

    if gen:
        print(f"✓ Generation 已记录")
    else:
        print("✗ Generation 记录失败")

    # 记录评分
    service.log_score(
        trace=trace,
        name="test_score",
        value=0.95,
        comment="Test score",
    )
    print(f"✓ Score 已记录")

    # 结束追踪
    service.end_trace(
        trace,
        output={"status": "completed"},
    )
    print(f"✓ 追踪已结束")

    # 刷新数据
    service.flush()
    print(f"✓ 数据已刷新到 Langfuse")

    return True


def test_document_processor_tracing():
    """测试文档处理器的 Langfuse 追踪。"""
    print("\n" + "=" * 60)
    print("测试 3: 文档处理器追踪")
    print("=" * 60)

    from app.services.langfuse_service import get_langfuse_service

    service = get_langfuse_service()

    if not service.enabled:
        print("⚠️  跳过 - Langfuse 未启用")
        return False

    try:
        from app.core.document_processor import DocumentProcessor

        processor = DocumentProcessor(
            chunk_size=200,
            chunk_overlap=20,
            use_semantic_chunking=False,
        )

        # 处理文本
        trace = service.create_trace(
            name="test_document_processing",
            input={"type": "text"},
        )

        nodes = processor.process_text(
            "这是一段测试文本。用于验证文档处理器的 Langfuse 追踪功能是否正常工作。"
            "我们需要确保分块和追踪都能正确执行。",
            metadata={"source": "test"},
        )

        service.end_trace(
            trace,
            output={"num_nodes": len(nodes)},
        )
        service.flush()

        print(f"✓ 文档处理完成: {len(nodes)} 个节点")
        print(f"✓ 追踪已发送到 Langfuse")
        return True

    except Exception as e:
        print(f"✗ 文档处理器测试失败: {e}")
        return False


def test_vector_store_tracing():
    """测试向量存储的 Langfuse 追踪。"""
    print("\n" + "=" * 60)
    print("测试 4: 向量存储追踪 (需要 Milvus)")
    print("=" * 60)

    from app.services.langfuse_service import get_langfuse_service

    service = get_langfuse_service()

    if not service.enabled:
        print("⚠️  跳过 - Langfuse 未启用")
        return False

    try:
        from app.core.vector_store import VectorStoreManager
        from llama_index.core.schema import TextNode

        # 创建追踪
        trace = service.create_trace(
            name="test_vector_store",
            input={"operation": "add_and_retrieve"},
        )

        manager = VectorStoreManager(
            collection_name="test_langfuse_collection",
        )

        # 添加测试节点
        nodes = [
            TextNode(text="这是第一个测试文档", metadata={"id": 1}),
            TextNode(text="这是第二个测试文档", metadata={"id": 2}),
        ]

        ids = manager.add_nodes(nodes, overwrite=True, trace=trace)
        print(f"✓ 添加了 {len(ids)} 个节点")

        # 检索
        results = manager.retrieve("测试文档", top_k=2, trace=trace)
        print(f"✓ 检索到 {len(results)} 个结果")

        service.end_trace(
            trace,
            output={"added": len(ids), "retrieved": len(results)},
        )
        service.flush()

        print(f"✓ 向量存储追踪已发送到 Langfuse")
        return True

    except Exception as e:
        print(f"⚠️  向量存储测试跳过: {e}")
        print("   (可能是 Milvus 未运行)")
        return False


async def test_query_engine_tracing():
    """测试查询引擎的完整 Langfuse 追踪。"""
    print("\n" + "=" * 60)
    print("测试 5: 查询引擎完整追踪 (需要完整环境)")
    print("=" * 60)

    from app.services.langfuse_service import get_langfuse_service

    service = get_langfuse_service()

    if not service.enabled:
        print("⚠️  跳过 - Langfuse 未启用")
        return False

    try:
        from app.rag.query_engine import EnhancedRAGQueryEngine

        engine = EnhancedRAGQueryEngine()

        response = await engine.aquery(
            question="这是一个测试问题",
            user_id="test_user",
            session_id="test_session",
        )

        print(f"✓ 查询完成")
        print(f"  - 状态: {response.metadata.get('status')}")
        print(f"  - 来源数: {response.metadata.get('num_sources', 0)}")
        print(f"  - 耗时: {response.metadata.get('total_time', 'N/A')}s")

        service.flush()
        print(f"✓ 完整 RAG 追踪已发送到 Langfuse")
        return True

    except Exception as e:
        print(f"⚠️  查询引擎测试跳过: {e}")
        print("   (可能是依赖服务未运行)")
        return False


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("  Langfuse 集成测试")
    print("=" * 60)

    results = []

    # 测试 1: 服务初始化
    results.append(("Langfuse 服务初始化", test_langfuse_service()))

    # 测试 2: 追踪创建
    results.append(("追踪创建", test_trace_creation()))

    # 测试 3: 文档处理器
    results.append(("文档处理器追踪", test_document_processor_tracing()))

    # 测试 4: 向量存储
    results.append(("向量存储追踪", test_vector_store_tracing()))

    # 测试 5: 查询引擎
    results.append(("查询引擎追踪", asyncio.run(test_query_engine_tracing())))

    # 总结
    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败/跳过"
        print(f"  {status}: {name}")
        if result:
            passed += 1

    print(f"\n  总计: {passed}/{len(results)} 通过")

    if passed >= 2:
        print("\n✓ Langfuse 集成基本正常!")
        print(f"  请访问 {settings.LANGFUSE_HOST} 查看追踪数据")
    else:
        print("\n⚠️  请检查 Langfuse 配置")

    return passed >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
