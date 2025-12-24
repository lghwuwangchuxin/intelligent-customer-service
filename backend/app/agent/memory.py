"""
Memory Management - Multi-turn conversation memory with summarization.
多轮对话记忆管理模块，支持自动摘要和上下文管理。

日志追踪结构
-----------
```
Memory Operations:
├── AddMessage - 添加消息到对话
│   ├── 会话ID、角色、消息计数
│   └── 自动触发摘要检查
├── Summarize - 对话内容摘要
│   ├── 开始摘要（消息数、保留数）
│   ├── LLM调用生成摘要
│   └── 完成（摘要长度、耗时）
├── GetContext - 获取上下文
│   ├── 消息数量、是否包含摘要
│   └── 返回格式化的消息列表
├── ClearMemory - 清空记忆
└── DeleteMemory - 删除记忆
```
"""
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# 增强日志工具
from app.utils.log_utils import (
    LogContext,
    MemoryLogger,
    log_step,
    log_substep,
    log_memory_operation,
    log_llm_call,
    SEPARATOR_LIGHT,
)

logger = logging.getLogger(__name__)

# 全局记忆日志记录器
_memory_logger = MemoryLogger()


class ConversationMemory(BaseModel):
    """Memory for a single conversation."""
    conversation_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow()

    def get_recent_messages(self, n: int = 10) -> List[Dict[str, str]]:
        """Get the most recent n messages."""
        return self.messages[-n:] if self.messages else []

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        Get messages formatted for context.
        Includes summary if available.
        """
        context = []

        # Add summary as system message if available
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}",
            })

        # Add recent messages
        context.extend(self.get_recent_messages())

        return context

    def clear(self) -> None:
        """Clear all messages but keep summary."""
        self.messages = []
        self.updated_at = datetime.utcnow()


class MemoryManager:
    """
    Manager for conversation memories with summarization support.

    Features:
    - Multi-conversation memory management
    - Automatic summarization of old messages
    - Memory persistence (optional)
    """

    SUMMARIZATION_PROMPT = """请将以下对话内容总结为简洁的要点，保留关键信息：

{messages}

总结要点："""

    def __init__(
        self,
        llm_manager=None,
        max_messages: int = 50,
        summary_threshold: int = 20,
        keep_recent: int = 10,
    ):
        """
        Initialize memory manager.

        Args:
            llm_manager: LLM manager for summarization.
            max_messages: Maximum messages before truncation.
            summary_threshold: Trigger summarization at this count.
            keep_recent: Number of recent messages to keep after summarization.
        """
        self.llm_manager = llm_manager
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self._memories: Dict[str, ConversationMemory] = {}

    def get_or_create(self, conversation_id: str) -> ConversationMemory:
        """Get existing memory or create new one."""
        if conversation_id not in self._memories:
            self._memories[conversation_id] = ConversationMemory(
                conversation_id=conversation_id
            )
        return self._memories[conversation_id]

    def get_memory(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Get memory for a conversation."""
        return self._memories.get(conversation_id)

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to conversation memory.
        Triggers summarization if threshold is reached.

        Args:
            conversation_id: Conversation identifier.
            role: Message role (user/assistant/system).
            content: Message content.
        """
        memory = self.get_or_create(conversation_id)
        memory.add_message(role, content)

        # 记录添加消息
        _memory_logger.log_add_message(conversation_id, role, len(memory.messages))

        # 检查是否需要触发摘要
        if len(memory.messages) >= self.summary_threshold:
            logger.info(f"[Memory] 会话 {conversation_id[:8]} 消息数 ({len(memory.messages)}) 达到阈值 ({self.summary_threshold})，触发摘要")
            await self._summarize(memory)

        logger.debug(
            f"[Memory] 添加消息到会话 {conversation_id[:8]} | "
            f"角色: {role} | 总消息: {len(memory.messages)} | "
            f"内容: {content[:50]}{'...' if len(content) > 50 else ''}"
        )

    async def _summarize(self, memory: ConversationMemory) -> None:
        """
        Summarize older messages in the memory.

        Args:
            memory: Conversation memory to summarize.
        """
        conversation_id = memory.conversation_id
        total_messages = len(memory.messages)

        if not self.llm_manager:
            # No LLM available, just truncate
            logger.warning(f"[Memory] 会话 {conversation_id[:8]} 无LLM管理器，直接截断消息")
            memory.messages = memory.messages[-self.keep_recent:]
            return

        # 记录摘要开始
        _memory_logger.log_summarize_start(conversation_id, total_messages, self.keep_recent)
        summarize_start = time.time()

        try:
            # Get messages to summarize (all except recent)
            to_summarize = memory.messages[:-self.keep_recent]

            if not to_summarize:
                logger.debug(f"[Memory] 会话 {conversation_id[:8]} 无需摘要的消息")
                return

            # Format messages for summarization
            formatted = "\n".join([
                f"{m['role']}: {m['content']}"
                for m in to_summarize
            ])

            logger.info(
                f"[Memory] 会话 {conversation_id[:8]} 开始摘要 | "
                f"待摘要: {len(to_summarize)} 条 | 保留: {self.keep_recent} 条 | "
                f"文本长度: {len(formatted)} 字符"
            )

            # Generate summary
            llm_start = time.time()
            prompt = self.SUMMARIZATION_PROMPT.format(messages=formatted)
            summary = await self.llm_manager.ainvoke([
                {"role": "user", "content": prompt}
            ])
            llm_elapsed = int((time.time() - llm_start) * 1000)

            logger.info(
                f"[Memory] 会话 {conversation_id[:8]} LLM摘要完成 | "
                f"输入: {len(formatted)} 字符 | 输出: {len(summary)} 字符 | "
                f"耗时: {llm_elapsed}ms"
            )

            # Update memory
            if memory.summary:
                # Combine with existing summary
                old_summary_len = len(memory.summary)
                memory.summary = f"{memory.summary}\n\n{summary}"
                logger.debug(
                    f"[Memory] 会话 {conversation_id[:8]} 合并摘要 | "
                    f"旧摘要: {old_summary_len} 字符 | 新摘要: {len(summary)} 字符"
                )
            else:
                memory.summary = summary

            # Keep only recent messages
            memory.messages = memory.messages[-self.keep_recent:]

            summarize_elapsed = int((time.time() - summarize_start) * 1000)

            # 记录摘要完成
            _memory_logger.log_summarize_complete(conversation_id, len(memory.summary), summarize_elapsed)

            logger.info(
                f"[Memory] 会话 {conversation_id[:8]} 摘要完成 ✓ | "
                f"摘要长度: {len(memory.summary)} 字符 | "
                f"保留消息: {len(memory.messages)} 条 | "
                f"总耗时: {summarize_elapsed}ms"
            )

        except Exception as e:
            summarize_elapsed = int((time.time() - summarize_start) * 1000)
            logger.error(
                f"[Memory] 会话 {conversation_id[:8]} 摘要失败 ✗ | "
                f"错误: {str(e)[:100]} | 耗时: {summarize_elapsed}ms"
            )
            # Fallback: just truncate
            memory.messages = memory.messages[-self.keep_recent:]
            logger.warning(f"[Memory] 会话 {conversation_id[:8]} 回退到截断模式，保留最近 {self.keep_recent} 条消息")

    def get_context(
        self,
        conversation_id: str,
        include_summary: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM.

        Args:
            conversation_id: Conversation identifier.
            include_summary: Whether to include summary.

        Returns:
            List of messages for context.
        """
        memory = self.get_memory(conversation_id)
        if not memory:
            logger.debug(f"[Memory] 会话 {conversation_id[:8]} 未找到记忆")
            return []

        if include_summary:
            context = memory.get_context_messages()
        else:
            context = memory.get_recent_messages(self.keep_recent)

        # 记录获取上下文
        _memory_logger.log_get_context(
            conversation_id,
            len(context),
            memory.summary is not None
        )

        logger.debug(
            f"[Memory] 获取上下文 | 会话: {conversation_id[:8]} | "
            f"消息数: {len(context)} | 包含摘要: {memory.summary is not None}"
        )

        return context

    def clear_memory(self, conversation_id: str) -> None:
        """Clear memory for a conversation."""
        if conversation_id in self._memories:
            memory = self._memories[conversation_id]
            msg_count = len(memory.messages)
            memory.clear()
            logger.info(
                f"[Memory] 清空记忆 | 会话: {conversation_id[:8]} | "
                f"清除消息: {msg_count} 条 | 保留摘要: {memory.summary is not None}"
            )

    def delete_memory(self, conversation_id: str) -> bool:
        """Delete memory for a conversation entirely."""
        if conversation_id in self._memories:
            memory = self._memories[conversation_id]
            msg_count = len(memory.messages)
            has_summary = memory.summary is not None
            del self._memories[conversation_id]
            logger.info(
                f"[Memory] 删除记忆 | 会话: {conversation_id[:8]} | "
                f"删除消息: {msg_count} 条 | 删除摘要: {has_summary}"
            )
            return True
        logger.debug(f"[Memory] 删除记忆失败 | 会话 {conversation_id[:8]} 不存在")
        return False

    def get_all_conversation_ids(self) -> List[str]:
        """Get all conversation IDs with memories."""
        ids = list(self._memories.keys())
        logger.debug(f"[Memory] 获取所有会话ID | 共 {len(ids)} 个会话")
        return ids

    def export_memory(self, conversation_id: str) -> Optional[Dict]:
        """Export memory as dict for persistence."""
        memory = self.get_memory(conversation_id)
        if memory:
            data = memory.model_dump()
            logger.info(
                f"[Memory] 导出记忆 | 会话: {conversation_id[:8]} | "
                f"消息数: {len(memory.messages)} | 有摘要: {memory.summary is not None}"
            )
            return data
        logger.debug(f"[Memory] 导出记忆失败 | 会话 {conversation_id[:8]} 不存在")
        return None

    def import_memory(self, data: Dict) -> ConversationMemory:
        """Import memory from dict."""
        memory = ConversationMemory(**data)
        self._memories[memory.conversation_id] = memory
        logger.info(
            f"[Memory] 导入记忆 | 会话: {memory.conversation_id[:8]} | "
            f"消息数: {len(memory.messages)} | 有摘要: {memory.summary is not None}"
        )
        return memory
