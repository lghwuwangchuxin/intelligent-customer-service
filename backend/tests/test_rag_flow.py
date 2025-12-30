"""
Complete RAG Flow Test
======================

测试 RAG 流程的完整性和性能优化效果。

包含测试:
1. 嵌入缓存测试
2. 批量嵌入测试
3. 后处理器测试
4. 混合检索测试
5. 完整查询流程测试
"""
import asyncio
import time
import logging
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_embedding_cache():
    """测试嵌入缓存功能"""
    print("\n" + "="*60)
    print("测试 1: 嵌入缓存功能")
    print("="*60)

    from app.rag.postprocessor import EmbeddingCache, get_embeddings_batch
    import numpy as np

    cache = EmbeddingCache(max_size=100)

    # Test cache set and get
    test_content = "这是一个测试文本"
    test_embedding = np.random.rand(768)

    cache.set(test_content, test_embedding)
    retrieved = cache.get(test_content)

    assert retrieved is not None, "缓存读取失败"
    assert np.allclose(test_embedding, retrieved), "缓存内容不匹配"
    print("✓ 缓存基本功能正常")

    # Test batch embedding with cache
    texts = [
        "如何申请退款？",
        "退款流程是什么？",
        "产品保修期是多久？",
    ]

    # First call - should compute embeddings
    start = time.time()
    embeddings1 = get_embeddings_batch(texts, use_cache=True)
    time1 = time.time() - start
    print(f"✓ 首次嵌入: {len(embeddings1)} 个, 耗时: {time1*1000:.0f}ms")

    # Second call - should use cache
    start = time.time()
    embeddings2 = get_embeddings_batch(texts, use_cache=True)
    time2 = time.time() - start
    print(f"✓ 缓存命中: {len(embeddings2)} 个, 耗时: {time2*1000:.0f}ms")

    assert time2 < time1, f"缓存应该更快: {time2:.3f}s vs {time1:.3f}s"
    print(f"✓ 缓存加速: {time1/time2:.1f}x 倍")

    return True


def test_batch_embedding():
    """测试批量嵌入性能"""
    print("\n" + "="*60)
    print("测试 2: 批量嵌入性能")
    print("="*60)

    from app.core.embeddings import get_embedding_manager

    embed_manager = get_embedding_manager()

    texts = [f"测试文档 {i}: 这是一段用于测试的文本内容。" for i in range(10)]

    # Test individual embedding
    start = time.time()
    individual_results = [embed_manager.embed_query(t) for t in texts]
    individual_time = time.time() - start
    print(f"✓ 逐个嵌入: {len(individual_results)} 个, 耗时: {individual_time*1000:.0f}ms")

    # Test batch embedding
    start = time.time()
    batch_results = embed_manager.embed_documents(texts)
    batch_time = time.time() - start
    print(f"✓ 批量嵌入: {len(batch_results)} 个, 耗时: {batch_time*1000:.0f}ms")

    if batch_time < individual_time:
        print(f"✓ 批量嵌入更快: {individual_time/batch_time:.1f}x 倍")
    else:
        print(f"⚠ 批量嵌入未显著加速 (可能是小批量)")

    return True


def test_postprocessor():
    """测试后处理器功能"""
    print("\n" + "="*60)
    print("测试 3: 后处理器功能")
    print("="*60)

    from llama_index.core.schema import TextNode, NodeWithScore
    from app.rag.postprocessor import (
        SemanticDeduplicator,
        MMRPostprocessor,
        RAGPostProcessor
    )
    from llama_index.core.schema import QueryBundle

    # Create test nodes with different content
    test_nodes = [
        NodeWithScore(
            node=TextNode(text="如何申请退款？需要提供订单号和退款原因。", id_="1"),
            score=0.95
        ),
        NodeWithScore(
            node=TextNode(text="退款申请流程：1.登录账户 2.找到订单 3.点击退款", id_="2"),
            score=0.90
        ),
        NodeWithScore(
            node=TextNode(text="如何申请退款？请提供订单编号和退款理由。", id_="3"),  # 与1相似
            score=0.85
        ),
        NodeWithScore(
            node=TextNode(text="产品保修期为一年，从购买日期开始计算。", id_="4"),
            score=0.70
        ),
        NodeWithScore(
            node=TextNode(text="客服工作时间为周一至周五 9:00-18:00。", id_="5"),
            score=0.60
        ),
    ]

    # Test deduplicator
    dedup = SemanticDeduplicator(similarity_threshold=0.90, enable_dedup=True)
    deduped = dedup.postprocess_nodes(test_nodes)
    print(f"✓ 语义去重: {len(test_nodes)} → {len(deduped)} 节点")

    # Test MMR
    mmr = MMRPostprocessor(lambda_mult=0.5, top_n=3)
    query = QueryBundle(query_str="如何申请退款？")
    mmr_result = mmr.postprocess_nodes(test_nodes, query)
    print(f"✓ MMR 多样性过滤: {len(test_nodes)} → {len(mmr_result)} 节点")

    # Test combined processor
    processor = RAGPostProcessor(
        enable_dedup=True,
        dedup_threshold=0.90,
        enable_mmr=True,
        mmr_lambda=0.5,
        final_top_k=3
    )
    final_result = processor.process(test_nodes, query="如何申请退款？")
    print(f"✓ 组合后处理: {len(test_nodes)} → {len(final_result)} 节点")

    return True


async def test_hybrid_retriever():
    """测试混合检索器"""
    print("\n" + "="*60)
    print("测试 4: 混合检索器")
    print("="*60)

    from app.rag.hybrid_retriever import HybridRetriever
    from llama_index.core.schema import TextNode, QueryBundle

    retriever = HybridRetriever(
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k=5,
        enable_hybrid=True
    )

    # Build BM25 index with test nodes
    test_nodes = [
        TextNode(text="如何申请退款？需要提供订单号和退款原因。", id_="1"),
        TextNode(text="退款流程：登录账户，找到订单，点击申请退款。", id_="2"),
        TextNode(text="产品保修期为一年，从购买日期开始计算。", id_="3"),
        TextNode(text="客服工作时间为周一至周五 9:00-18:00。", id_="4"),
        TextNode(text="如需咨询，请拨打客服热线 400-xxx-xxxx。", id_="5"),
    ]

    retriever.build_bm25_index(test_nodes)
    print(f"✓ BM25 索引构建完成: {len(test_nodes)} 个节点")

    # Test sync retrieval
    queries = ["如何申请退款", "保修期多长"]
    results = retriever.retrieve_multi_query(queries)
    print(f"✓ 同步多查询检索: {len(results)} 个结果")

    # Test async retrieval
    start = time.time()
    async_results = await retriever.aretrieve_multi_query(queries)
    async_time = time.time() - start
    print(f"✓ 异步多查询检索: {len(async_results)} 个结果, 耗时: {async_time*1000:.0f}ms")

    return True


async def test_full_query_flow():
    """测试完整查询流程"""
    print("\n" + "="*60)
    print("测试 5: 完整 RAG 查询流程")
    print("="*60)

    try:
        from app.rag.query_engine import EnhancedRAGQueryEngine, get_query_engine

        # Get query engine
        engine = get_query_engine()
        print("✓ RAG 查询引擎初始化成功")

        # Test query (requires indexed documents)
        question = "如何申请退款？"

        print(f"执行查询: {question}")
        start = time.time()

        response = await engine.aquery(
            question=question,
            user_id="test_user",
            session_id="test_session"
        )

        elapsed = time.time() - start

        print(f"✓ 查询完成, 耗时: {elapsed*1000:.0f}ms")
        print(f"  状态: {response.metadata.get('status', 'unknown')}")
        print(f"  来源数: {len(response.sources)}")
        print(f"  回答长度: {len(response.answer)} 字符")

        if response.answer:
            print(f"  回答预览: {response.answer[:100]}...")

        return True

    except Exception as e:
        print(f"⚠ 完整流程测试失败: {e}")
        print("  (可能需要先运行索引任务)")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("RAG 流程优化验证测试")
    print("="*60)
    print(f"LLM 模型: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
    print(f"嵌入模型: {settings.EMBEDDING_MODEL}")
    print(f"批次大小: {settings.EMBEDDING_BATCH_SIZE}")
    print(f"最大并发: {settings.EMBEDDING_MAX_CONCURRENT}")

    results = {}

    # Test 1: Embedding cache
    try:
        results['cache'] = test_embedding_cache()
    except Exception as e:
        print(f"✗ 嵌入缓存测试失败: {e}")
        results['cache'] = False

    # Test 2: Batch embedding
    try:
        results['batch'] = test_batch_embedding()
    except Exception as e:
        print(f"✗ 批量嵌入测试失败: {e}")
        results['batch'] = False

    # Test 3: Postprocessor
    try:
        results['postproc'] = test_postprocessor()
    except Exception as e:
        print(f"✗ 后处理器测试失败: {e}")
        results['postproc'] = False

    # Test 4: Hybrid retriever
    try:
        results['retriever'] = await test_hybrid_retriever()
    except Exception as e:
        print(f"✗ 混合检索器测试失败: {e}")
        results['retriever'] = False

    # Test 5: Full query flow
    try:
        results['full'] = await test_full_query_flow()
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        results['full'] = False

    # Summary
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
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
