"""
LLM Manager / 大语言模型管理器
==============================

统一的多 LLM 提供商接口，同时支持 LlamaIndex 和 LangChain。

支持的提供商
-----------
- Ollama (本地模型)
- OpenAI (GPT 系列)
- Anthropic Claude
- DeepSeek
- 阿里云通义千问
- 百度文心一言
- 火山引擎豆包
- 智谱 AI
- 月之暗面 Kimi

架构说明
--------
```
LLMManager
    ├── LlamaIndex LLM (用于 RAG 管道)
    │   └── Ollama / OpenAILike
    │
    └── LangChain LLM (用于 )
        └── ChatOllama / ChatOpenAI / ChatAnthropic
```

模型管理功能
-----------
- 动态模型切换
- 连接验证和健康检查
- Ollama 本地模型列表获取
- 提供商配置验证

性能优化
--------
- LLM 调用超时控制 (默认 120 秒)
- 自动重试机制 (最多 3 次，指数退避)
- 请求延迟监控和日志记录

Langfuse 追踪
-------------
```
Generation: llm_call
├── model
├── provider
├── input (messages)
├── output (response)
├── usage (tokens)
└── latency
```

配置参数
--------
```python
# config/settings.py
LLM_PROVIDER = "ollama"
LLM_MODEL = "qwen3:latest"
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 120  # 超时时间（秒）
LLM_MAX_RETRIES = 3  # 最大重试次数
LANGFUSE_ENABLED = True
```

使用示例
--------
```python
from app.core.llm_manager import LLMManager

# 初始化
manager = LLMManager(provider="ollama", model="qwen3:latest")

# 同步调用
response = manager.invoke([{"role": "user", "content": "你好"}])

# 异步调用（带超时和重试）
response = await manager.ainvoke(
    messages=[{"role": "user", "content": "你好"}],
    timeout=60.0,  # 自定义超时
    max_retries=2  # 自定义重试次数
)

# 流式调用
for chunk in manager.stream([{"role": "user", "content": "你好"}]):
    print(chunk, end="")

# 测试模型连接
result = await manager.test_connection()
if result["success"]:
    print("连接成功")

# 获取 Ollama 模型列表
models = await LLMManager.fetch_ollama_models("http://localhost:11434")
```

Author: Intelligent Customer Service Team
Version: 2.3.0 (添加超时和重试机制)
"""
import logging
import time
import asyncio
import aiohttp
from typing import Iterator, List, Dict, Any, Optional, Union, AsyncIterator

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOpenAI

# LlamaIndex imports
from llama_index.core import Settings as LlamaSettings
from llama_index.llms.ollama import Ollama as LlamaOllama

# Langfuse observability
from app.services.langfuse_service import get_langfuse_service

# Try to import OpenAI-like LLM for LlamaIndex
try:
    from llama_index.llms.openai_like import OpenAILike as LlamaOpenAILike
    LLAMA_OPENAI_LIKE_AVAILABLE = True
except ImportError:
    LLAMA_OPENAI_LIKE_AVAILABLE = False
    LlamaOpenAILike = None

# Claude support - optional import (LangChain)
try:
    from langchain_anthropic import ChatAnthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ChatAnthropic = None

# Claude support - LlamaIndex
try:
    from llama_index.llms.anthropic import Anthropic as LlamaAnthropic
    LLAMA_ANTHROPIC_AVAILABLE = True
except ImportError:
    LLAMA_ANTHROPIC_AVAILABLE = False
    LlamaAnthropic = None

logger = logging.getLogger(__name__)


# ==================== 默认配置 ====================

DEFAULT_LLM_TIMEOUT = 300.0  # 默认超时时间（秒）- 本地 LLM 可能较慢
DEFAULT_MAX_RETRIES = 3  # 默认最大重试次数
DEFAULT_RETRY_DELAY = 1.0  # 默认重试延迟（秒）
RETRYABLE_ERRORS = (
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


# ==================== 提供商配置 ====================

PROVIDER_CONFIGS = {
    "ollama": {
        "name": "Ollama (本地模型)",
        "base_url": "http://localhost:11434",
        "models": ["qwen3:latest", "llama3:latest", "deepseek-r1:7b", "mistral:latest"],
        "requires_api_key": False,
    },
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "requires_api_key": True,
    },
    "claude": {
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
        "requires_api_key": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "requires_api_key": True,
    },
    "aliyun": {
        "name": "阿里云 (通义千问)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
        "requires_api_key": True,
    },
    "baidu": {
        "name": "百度 (文心一言)",
        "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
        "models": ["ernie-4.0-8k", "ernie-3.5-8k", "ernie-speed-8k"],
        "requires_api_key": True,
    },
    "volcengine": {
        "name": "火山引擎 (豆包)",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "models": ["doubao-pro-32k", "doubao-lite-32k"],
        "requires_api_key": True,
    },
    "zhipu": {
        "name": "智谱 AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-plus", "glm-4", "glm-4-flash"],
        "requires_api_key": True,
    },
    "moonshot": {
        "name": "月之暗面 (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "requires_api_key": True,
    },
}


class LLMManager:
    """
    统一 LLM 管理器 (LlamaIndex + LangChain)
    ======================================

    支持多个 LLM 提供商，同时提供 LlamaIndex 和 LangChain 两种接口。

    Attributes
    ----------
    provider : str
        LLM 提供商名称

    model : str
        模型名称

    temperature : float
        生成温度

    max_tokens : int
        最大生成长度

    _llm : ChatOllama | ChatOpenAI | ChatAnthropic
        LangChain LLM 实例

    _llama_llm : LlamaOllama | LlamaOpenAILike
        LlamaIndex LLM 实例

    Example
    -------
    ```python
    manager = LLMManager(
        provider="ollama",
        model="qwen3:latest",
        temperature=0.7
    )

    # 普通调用
    response = manager.invoke([
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ])

    # 流式调用
    for chunk in manager.stream(messages):
        print(chunk, end="")

    # 获取 LlamaIndex LLM
    llm = manager.get_llama_llm()
    response = llm.complete("你好")
    ```
    """

    PROVIDER_BASE_URLS = {
        "ollama": "http://localhost:11434",
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "moonshot": "https://api.moonshot.cn/v1",
        "claude": "https://api.anthropic.com",
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        extended_thinking: bool = False,
        thinking_budget: int = 10000,
    ):
        """
        初始化 LLM 管理器。

        Parameters
        ----------
        provider : str
            提供商: ollama, openai, claude, deepseek 等

        model : str
            模型名称

        base_url : str, optional
            API 地址

        api_key : str, optional
            API 密钥

        temperature : float
            生成温度 (0.0 ~ 1.0)

        max_tokens : int
            最大生成长度

        extended_thinking : bool
            是否启用扩展思考 (Claude)

        thinking_budget : int
            思考预算 token 数
        """
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url or self.PROVIDER_BASE_URLS.get(self.provider)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extended_thinking = extended_thinking
        self.thinking_budget = thinking_budget
        self._tools: List[BaseTool] = []

        # 初始化 LLM
        self._llm = self._init_langchain_llm()
        self._llama_llm = self._init_llama_llm()
        self._llm_with_tools = None

        # 设置 LlamaIndex 全局 LLM
        if self._llama_llm:
            LlamaSettings.llm = self._llama_llm

        logger.info(
            f"[LLM] Initialized - provider: {self.provider}, "
            f"model: {self.model}, temperature: {self.temperature}"
        )

    def _init_langchain_llm(self):
        """初始化 LangChain LLM。"""
        if self.provider == "ollama":
            return ChatOllama(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
        elif self.provider == "claude":
            if not CLAUDE_AVAILABLE:
                raise ImportError(
                    "langchain-anthropic is required for Claude. "
                    "Install with: pip install langchain-anthropic"
                )
            if not self.api_key:
                raise ValueError("CLAUDE_API_KEY is required for Claude provider")

            kwargs = {
                "model": self.model,
                "anthropic_api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            if self.extended_thinking:
                kwargs["model_kwargs"] = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget,
                    }
                }
                kwargs["temperature"] = 1.0

            return ChatAnthropic(**kwargs)
        else:
            # OpenAI-compatible providers
            return ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    def _init_llama_llm(self):
        """
        初始化 LlamaIndex LLM。

        用于 RAG 管道中的响应生成。
        """
        try:
            if self.provider == "ollama":
                return LlamaOllama(
                    model=self.model,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    request_timeout=300.0,  # 本地 LLM 可能较慢，增加超时时间
                )
            elif self.provider == "claude" and LLAMA_ANTHROPIC_AVAILABLE:
                logger.info(f"[LLM] Initializing LlamaIndex Anthropic: {self.model}")
                return LlamaAnthropic(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            elif LLAMA_OPENAI_LIKE_AVAILABLE and self.provider in (
                "openai", "deepseek", "zhipu", "moonshot", "aliyun"
            ):
                return LlamaOpenAILike(
                    model=self.model,
                    api_base=self.base_url,
                    api_key=self.api_key or "not-needed",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            else:
                logger.warning(
                    f"[LLM] LlamaIndex LLM not available for {self.provider}, "
                    "using LangChain only"
                )
                return None
        except Exception as e:
            logger.warning(f"[LLM] Failed to init LlamaIndex LLM: {e}")
            return None

    @property
    def llm(self):
        """获取 LangChain LLM 实例。"""
        return self._llm

    def get_llama_llm(self):
        """
        获取 LlamaIndex LLM 实例。

        用于 RAG 管道中的响应生成。

        Returns
        -------
        LlamaOllama | LlamaOpenAILike | None
            LlamaIndex LLM 实例
        """
        return self._llama_llm

    def invoke(
        self,
        messages: List[Dict[str, str]],
        trace=None,
    ) -> str:
        """
        同步调用 LLM。

        Parameters
        ----------
        messages : List[Dict[str, str]]
            消息列表，每个消息包含 'role' 和 'content'

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        Returns
        -------
        str
            LLM 响应内容

        Example
        -------
        ```python
        response = manager.invoke([
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"}
        ])
        ```
        """
        logger.info(
            f"[LLM] Invoke - provider: {self.provider}/{self.model}, "
            f"messages: {len(messages)}"
        )

        start_time = time.time()
        langchain_messages = self._convert_messages(messages)
        response = self._llm.invoke(langchain_messages)
        elapsed = time.time() - start_time

        logger.info(f"[LLM] Invoke complete - response length: {len(response.content)}")

        # Langfuse 追踪
        langfuse = get_langfuse_service()
        if trace and langfuse.enabled:
            # 估算 token 使用量 (简单估算: 1 token ≈ 4 字符)
            input_text = " ".join(m.get("content", "") for m in messages)
            input_tokens = len(input_text) // 4
            output_tokens = len(response.content) // 4

            langfuse.log_generation(
                trace=trace,
                name="llm_invoke",
                model=f"{self.provider}/{self.model}",
                input=messages,
                output=response.content,
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                metadata={
                    "provider": self.provider,
                    "temperature": self.temperature,
                    "latency_seconds": round(elapsed, 3),
                },
            )

        return response.content

    async def ainvoke(
        self,
        messages: List[Dict[str, str]],
        trace=None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        异步调用 LLM（带超时和重试机制）。

        Parameters
        ----------
        messages : List[Dict[str, str]]
            消息列表

        trace : Langfuse Trace, optional
            Langfuse 追踪对象

        timeout : float, optional
            超时时间（秒），默认 120 秒

        max_retries : int, optional
            最大重试次数，默认 3 次

        Returns
        -------
        str
            LLM 响应内容

        Raises
        ------
        asyncio.TimeoutError
            如果所有重试都超时

        Exception
            如果所有重试都失败
        """
        timeout = timeout or DEFAULT_LLM_TIMEOUT
        max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

        logger.info(
            f"[LLM] Async invoke - provider: {self.provider}/{self.model}, "
            f"messages: {len(messages)}, timeout: {timeout}s, max_retries: {max_retries}"
        )

        start_time = time.time()
        langchain_messages = self._convert_messages(messages)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # 带超时的 LLM 调用
                response = await asyncio.wait_for(
                    self._llm.ainvoke(langchain_messages),
                    timeout=timeout
                )
                elapsed = time.time() - start_time

                logger.info(
                    f"[LLM] Async invoke complete - response length: {len(response.content)}, "
                    f"elapsed: {elapsed:.2f}s, attempt: {attempt + 1}"
                )

                # Langfuse 追踪
                langfuse = get_langfuse_service()
                if trace and langfuse.enabled:
                    input_text = " ".join(m.get("content", "") for m in messages)
                    input_tokens = len(input_text) // 4
                    output_tokens = len(response.content) // 4

                    langfuse.log_generation(
                        trace=trace,
                        name="llm_ainvoke",
                        model=f"{self.provider}/{self.model}",
                        input=messages,
                        output=response.content,
                        usage={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                        metadata={
                            "provider": self.provider,
                            "temperature": self.temperature,
                            "latency_seconds": round(elapsed, 3),
                            "attempt": attempt + 1,
                        },
                    )

                return response.content

            except asyncio.TimeoutError as e:
                last_error = e
                elapsed = time.time() - start_time
                logger.warning(
                    f"[LLM] Timeout after {elapsed:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                if attempt < max_retries:
                    # 指数退避重试
                    delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"[LLM] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

            except RETRYABLE_ERRORS as e:
                last_error = e
                elapsed = time.time() - start_time
                logger.warning(
                    f"[LLM] Retryable error: {type(e).__name__}: {e} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                if attempt < max_retries:
                    delay = DEFAULT_RETRY_DELAY * (2 ** attempt)
                    logger.info(f"[LLM] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

            except Exception as e:
                # 不可重试的错误，直接抛出
                elapsed = time.time() - start_time
                logger.error(
                    f"[LLM] Non-retryable error after {elapsed:.2f}s: {type(e).__name__}: {e}"
                )
                raise

        # 所有重试都失败
        elapsed = time.time() - start_time
        logger.error(
            f"[LLM] All {max_retries + 1} attempts failed after {elapsed:.2f}s"
        )
        raise last_error or asyncio.TimeoutError(f"LLM call failed after {max_retries + 1} attempts")

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """
        流式调用 LLM。

        Yields
        ------
        str
            响应内容片段

        Example
        -------
        ```python
        for chunk in manager.stream(messages):
            print(chunk, end="", flush=True)
        ```
        """
        logger.info(f"[LLM] Stream - provider: {self.provider}/{self.model}")
        langchain_messages = self._convert_messages(messages)
        chunk_count = 0
        for chunk in self._llm.stream(langchain_messages):
            if chunk.content:
                chunk_count += 1
                yield chunk.content
        logger.info(f"[LLM] Stream complete - chunks: {chunk_count}")

    async def astream(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        """异步流式调用 LLM。"""
        logger.info(f"[LLM] Async stream - provider: {self.provider}/{self.model}")
        langchain_messages = self._convert_messages(messages)
        chunk_count = 0
        async for chunk in self._llm.astream(langchain_messages):
            if chunk.content:
                chunk_count += 1
                yield chunk.content
        logger.info(f"[LLM] Async stream complete - chunks: {chunk_count}")

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List:
        """将字典消息转换为 LangChain 消息对象。"""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result

    def get_info(self) -> Dict[str, Any]:
        """获取 LLM 配置信息。"""
        info = {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llama_llm_available": self._llama_llm is not None,
        }
        if self.provider == "claude":
            info["extended_thinking"] = self.extended_thinking
            info["tools_count"] = len(self._tools)
        return info

    def bind_tools(self, tools: List[Union[BaseTool, Dict[str, Any]]]) -> "LLMManager":
        """
        绑定工具到 LLM（用于函数调用）。

        Parameters
        ----------
        tools : List
            LangChain 工具列表

        Returns
        -------
        LLMManager
            self，支持链式调用
        """
        self._tools = tools
        if self._tools:
            try:
                self._llm_with_tools = self._llm.bind_tools(tools)
                logger.info(f"[LLM] Bound {len(tools)} tools")
            except Exception as e:
                logger.warning(f"[LLM] Failed to bind tools: {e}")
                self._llm_with_tools = None
        return self

    def get_llm_with_tools(self):
        """获取绑定了工具的 LLM 实例。"""
        return self._llm_with_tools or self._llm

    @property
    def supports_tool_calling(self) -> bool:
        """检查当前提供商是否支持工具调用。"""
        return self.provider in ("claude", "openai", "deepseek")

    @classmethod
    def get_available_providers(cls) -> List[Dict[str, Any]]:
        """获取可用的 LLM 提供商列表。"""
        providers = []
        for provider_id, config in PROVIDER_CONFIGS.items():
            providers.append({
                "id": provider_id,
                "name": config["name"],
                "models": config["models"],
                "requires_api_key": config["requires_api_key"],
                "base_url": config["base_url"],
            })
        return providers

    @classmethod
    def get_provider_models(cls, provider: str) -> List[str]:
        """获取指定提供商的可用模型列表。"""
        config = PROVIDER_CONFIGS.get(provider.lower())
        if config:
            return config["models"]
        return []

    @classmethod
    def validate_provider_config(
        cls, provider: str, api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """验证提供商配置。"""
        config = PROVIDER_CONFIGS.get(provider.lower())
        if not config:
            return {"valid": False, "error": f"Unknown provider: {provider}"}

        if config["requires_api_key"] and not api_key:
            return {"valid": False, "error": f"{config['name']} requires an API key"}

        return {"valid": True, "provider": provider, "name": config["name"]}

    # ==================== 模型管理增强 ====================

    async def test_connection(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        测试模型连接和可用性。

        发送简单测试消息验证模型是否正常工作。

        Parameters
        ----------
        timeout : float
            超时时间（秒）

        Returns
        -------
        dict
            包含 success, latency, error 等信息
        """
        start_time = time.time()
        try:
            # 发送简单测试消息
            response = await asyncio.wait_for(
                self.ainvoke([{"role": "user", "content": "Hi"}]),
                timeout=timeout,
            )
            latency = time.time() - start_time

            return {
                "success": True,
                "provider": self.provider,
                "model": self.model,
                "latency": round(latency, 3),
                "response_preview": response[:50] if response else "",
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "provider": self.provider,
                "model": self.model,
                "error": f"Connection timeout after {timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "provider": self.provider,
                "model": self.model,
                "error": str(e),
            }

    @staticmethod
    async def fetch_ollama_models(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """
        获取 Ollama 本地可用模型列表。

        Parameters
        ----------
        base_url : str
            Ollama API 地址

        Returns
        -------
        dict
            包含 success, models 或 error
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model in data.get("models", []):
                            name = model.get("name", "")
                            size_bytes = model.get("size", 0)
                            size_gb = round(size_bytes / (1024**3), 2)
                            models.append({
                                "name": name,
                                "size": f"{size_gb} GB",
                                "size_bytes": size_bytes,
                                "modified": model.get("modified_at", ""),
                                "family": model.get("details", {}).get("family", ""),
                                "parameters": model.get("details", {}).get("parameter_size", ""),
                            })
                        return {
                            "success": True,
                            "models": models,
                            "count": len(models),
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}",
                        }
        except aiohttp.ClientConnectorError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Is it running?",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @staticmethod
    async def check_ollama_health(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
        """
        检查 Ollama 服务健康状态。

        Parameters
        ----------
        base_url : str
            Ollama API 地址

        Returns
        -------
        dict
            包含 healthy, version 等信息
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/api/version",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "version": data.get("version", "unknown"),
                        }
                    else:
                        return {"healthy": False, "error": f"HTTP {response.status}"}
        except aiohttp.ClientConnectorError:
            return {"healthy": False, "error": "Cannot connect to Ollama"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def switch_model(
        self,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> "LLMManager":
        """
        切换到新模型。

        Parameters
        ----------
        model : str
            新模型名称

        provider : str, optional
            新提供商（默认使用当前提供商）

        api_key : str, optional
            API 密钥

        base_url : str, optional
            API 地址

        Returns
        -------
        LLMManager
            self，支持链式调用
        """
        new_provider = provider or self.provider
        new_base_url = base_url or self.PROVIDER_BASE_URLS.get(new_provider) or self.base_url
        new_api_key = api_key or self.api_key

        logger.info(f"[LLM] Switching model: {self.provider}/{self.model} -> {new_provider}/{model}")

        # 验证配置
        validation = self.validate_provider_config(new_provider, new_api_key)
        if not validation.get("valid"):
            raise ValueError(validation.get("error"))

        # 更新配置
        self.provider = new_provider
        self.model = model
        self.base_url = new_base_url
        self.api_key = new_api_key

        # 重新初始化 LLM
        self._llm = self._init_langchain_llm()
        self._llama_llm = self._init_llama_llm()
        self._llm_with_tools = None

        # 更新 LlamaIndex 全局 LLM
        if self._llama_llm:
            LlamaSettings.llm = self._llama_llm

        logger.info(f"[LLM] Switched to {self.provider}/{self.model}")
        return self

    @classmethod
    async def test_provider_connection(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        测试提供商连接（不修改当前配置）。

        Parameters
        ----------
        provider : str
            提供商名称

        model : str
            模型名称

        api_key : str, optional
            API 密钥

        base_url : str, optional
            API 地址

        timeout : float
            超时时间

        Returns
        -------
        dict
            测试结果
        """
        # 验证配置
        validation = cls.validate_provider_config(provider, api_key)
        if not validation.get("valid"):
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "error": validation.get("error"),
            }

        try:
            # 创建临时 LLM 实例进行测试
            config = PROVIDER_CONFIGS.get(provider.lower(), {})
            test_url = base_url or config.get("base_url")

            test_llm = cls(
                provider=provider,
                model=model,
                base_url=test_url,
                api_key=api_key,
                temperature=0.1,
                max_tokens=50,
            )

            result = await test_llm.test_connection(timeout=timeout)
            return result

        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "error": str(e),
            }

    @classmethod
    async def get_all_available_models(cls, include_ollama: bool = True) -> Dict[str, Any]:
        """
        获取所有可用模型（包括 Ollama 本地模型）。

        Parameters
        ----------
        include_ollama : bool
            是否包含 Ollama 本地模型

        Returns
        -------
        dict
            按提供商分组的模型列表
        """
        result = {}

        # 添加预配置的提供商模型
        for provider_id, config in PROVIDER_CONFIGS.items():
            result[provider_id] = {
                "name": config["name"],
                "requires_api_key": config["requires_api_key"],
                "models": config["models"],
            }

        # 获取 Ollama 实际可用模型
        if include_ollama:
            ollama_result = await cls.fetch_ollama_models()
            if ollama_result.get("success"):
                ollama_models = [m["name"] for m in ollama_result.get("models", [])]
                result["ollama"]["models"] = ollama_models
                result["ollama"]["models_detail"] = ollama_result.get("models", [])

        return result

    async def ainvoke_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Union[BaseTool, Dict]]] = None,
    ) -> Dict[str, Any]:
        """
        带工具调用的异步 LLM 调用。

        Returns
        -------
        dict
            包含 'content' 和可选的 'tool_calls'
        """
        langchain_messages = self._convert_messages(messages)

        llm = self._llm
        if tools:
            llm = self._llm.bind_tools(tools)
        elif self._llm_with_tools:
            llm = self._llm_with_tools

        response = await llm.ainvoke(langchain_messages)
        result = {"content": response.content}

        if hasattr(response, "tool_calls") and response.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                }
                for tc in response.tool_calls
            ]

        return result

    async def astream_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Union[BaseTool, Dict]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """带工具调用的异步流式 LLM 调用。"""
        langchain_messages = self._convert_messages(messages)

        llm = self._llm
        if tools:
            llm = self._llm.bind_tools(tools)
        elif self._llm_with_tools:
            llm = self._llm_with_tools

        async for chunk in llm.astream(langchain_messages):
            if chunk.content:
                yield {"type": "content", "content": chunk.content}

            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    yield {
                        "type": "tool_call",
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", ""),
                    }


# ==================== 全局实例 ====================

_llm_manager: Optional[LLMManager] = None


def get_llm_manager(
    provider: str = None,
    model: str = None,
    **kwargs,
) -> LLMManager:
    """
    获取全局 LLM 管理器实例。

    Parameters
    ----------
    provider : str, optional
        LLM 提供商

    model : str, optional
        模型名称

    **kwargs
        其他参数

    Returns
    -------
    LLMManager
        LLM 管理器实例
    """
    global _llm_manager
    if _llm_manager is None:
        from config.settings import settings

        _llm_manager = LLMManager(
            provider=provider or settings.LLM_PROVIDER,
            model=model or settings.LLM_MODEL,
            base_url=kwargs.get("base_url") or settings.LLM_BASE_URL,
            api_key=kwargs.get("api_key") or settings.LLM_API_KEY,
            temperature=kwargs.get("temperature", settings.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", settings.LLM_MAX_TOKENS),
        )
    return _llm_manager
