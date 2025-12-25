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
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
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


class AgentInteraction(BaseModel):
    """Record of a single agent interaction (question-answer pair)."""
    interaction_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    question: str
    response: str
    thoughts: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


class ConversationMemory(BaseModel):
    """Memory for a single conversation."""
    conversation_id: str
    title: Optional[str] = None  # Auto-generated from first message
    messages: List[Dict[str, str]] = Field(default_factory=list)
    interactions: List[AgentInteraction] = Field(default_factory=list)  # Detailed interaction history
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

    def generate_title(self) -> str:
        """Generate a title from the first user message."""
        for msg in self.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Take first 50 chars as title
                title = content[:50].strip()
                if len(content) > 50:
                    title += "..."
                return title
        return f"对话 {self.conversation_id[:8]}"

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
        persist_path: Optional[str] = None,
    ):
        """
        Initialize memory manager.

        Args:
            llm_manager: LLM manager for summarization.
            max_messages: Maximum messages before truncation.
            summary_threshold: Trigger summarization at this count.
            keep_recent: Number of recent messages to keep after summarization.
            persist_path: Optional path for persisting memories to disk.
        """
        self.llm_manager = llm_manager
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent
        self._memories: Dict[str, ConversationMemory] = {}

        # Setup persistence
        self.persist_path: Optional[Path] = None
        if persist_path:
            self.persist_path = Path(persist_path)
            self.persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Memory] 持久化存储路径: {self.persist_path}")

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

    async def add_interaction(
        self,
        conversation_id: str,
        interaction_id: str,
        question: str,
        response: str,
        thoughts: List[Dict[str, Any]] = None,
        tool_calls: List[Dict[str, Any]] = None,
        iterations: int = 0,
        duration_ms: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """
        Add a complete agent interaction to memory.

        Args:
            conversation_id: Conversation identifier.
            interaction_id: Unique interaction ID.
            question: User's question.
            response: Agent's response.
            thoughts: List of thinking steps.
            tool_calls: List of tool calls made.
            iterations: Number of reasoning iterations.
            duration_ms: Total duration in milliseconds.
            error: Error message if any.
        """
        memory = self.get_or_create(conversation_id)

        interaction = AgentInteraction(
            interaction_id=interaction_id,
            question=question,
            response=response,
            thoughts=thoughts or [],
            tool_calls=tool_calls or [],
            iterations=iterations,
            duration_ms=duration_ms,
            error=error,
        )

        memory.interactions.append(interaction)
        memory.updated_at = datetime.utcnow()

        # Auto-generate title from first interaction
        if not memory.title and question:
            memory.title = question[:50].strip()
            if len(question) > 50:
                memory.title += "..."

        logger.info(
            f"[Memory] 添加交互记录 | 会话: {conversation_id[:8]} | "
            f"交互ID: {interaction_id[:8]} | 迭代: {iterations} | "
            f"工具调用: {len(tool_calls or [])} | 耗时: {duration_ms}ms"
        )

        # Persist if enabled
        if self.persist_path:
            await self._persist_memory(memory)

    def get_interactions(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AgentInteraction]:
        """
        Get interaction history for a conversation.

        Args:
            conversation_id: Conversation identifier.
            limit: Maximum number of interactions to return.
            offset: Number of interactions to skip.

        Returns:
            List of interactions.
        """
        memory = self.get_memory(conversation_id)
        if not memory:
            return []

        interactions = memory.interactions[offset:offset + limit]
        logger.debug(
            f"[Memory] 获取交互历史 | 会话: {conversation_id[:8]} | "
            f"返回: {len(interactions)} 条 | 总计: {len(memory.interactions)} 条"
        )
        return interactions

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        descending: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        List all conversations with summary info.

        Args:
            limit: Maximum number of conversations to return.
            offset: Number of conversations to skip.
            sort_by: Field to sort by (created_at, updated_at).
            descending: Sort in descending order.

        Returns:
            List of conversation summaries.
        """
        conversations = []

        for conv_id, memory in self._memories.items():
            conversations.append({
                "conversation_id": conv_id,
                "title": memory.title or memory.generate_title(),
                "message_count": len(memory.messages),
                "interaction_count": len(memory.interactions),
                "has_summary": memory.summary is not None,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "last_message": memory.messages[-1] if memory.messages else None,
            })

        # Sort
        if sort_by in ("created_at", "updated_at"):
            conversations.sort(
                key=lambda x: x[sort_by],
                reverse=descending,
            )

        # Paginate
        result = conversations[offset:offset + limit]

        logger.info(
            f"[Memory] 列出对话 | 返回: {len(result)} 条 | "
            f"总计: {len(conversations)} 条 | 排序: {sort_by} {'DESC' if descending else 'ASC'}"
        )

        return result

    def get_conversation_detail(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed conversation information.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Detailed conversation data or None.
        """
        memory = self.get_memory(conversation_id)
        if not memory:
            return None

        return {
            "conversation_id": conversation_id,
            "title": memory.title or memory.generate_title(),
            "messages": memory.messages,
            "interactions": [
                {
                    "interaction_id": i.interaction_id,
                    "timestamp": i.timestamp.isoformat(),
                    "question": i.question,
                    "response": i.response,
                    "thoughts": i.thoughts,
                    "tool_calls": i.tool_calls,
                    "iterations": i.iterations,
                    "duration_ms": i.duration_ms,
                    "error": i.error,
                }
                for i in memory.interactions
            ],
            "summary": memory.summary,
            "message_count": len(memory.messages),
            "interaction_count": len(memory.interactions),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "metadata": memory.metadata,
        }

    async def _persist_memory(self, memory: ConversationMemory) -> None:
        """Persist memory to disk."""
        if not self.persist_path:
            return

        try:
            file_path = self.persist_path / f"{memory.conversation_id}.json"
            data = memory.model_dump(mode="json")

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"[Memory] 持久化记忆 | 会话: {memory.conversation_id[:8]} | 文件: {file_path}")
        except Exception as e:
            logger.error(f"[Memory] 持久化失败 | 会话: {memory.conversation_id[:8]} | 错误: {e}")

    async def load_persisted_memories(self) -> int:
        """Load all persisted memories from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return 0

        loaded = 0
        for file_path in self.persist_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                memory = ConversationMemory(**data)
                self._memories[memory.conversation_id] = memory
                loaded += 1
            except Exception as e:
                logger.error(f"[Memory] 加载记忆失败 | 文件: {file_path} | 错误: {e}")

        logger.info(f"[Memory] 加载持久化记忆完成 | 加载: {loaded} 条")
        return loaded

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the title of a conversation."""
        memory = self.get_memory(conversation_id)
        if not memory:
            return False

        memory.title = title
        memory.updated_at = datetime.utcnow()
        logger.info(f"[Memory] 更新对话标题 | 会话: {conversation_id[:8]} | 新标题: {title[:30]}")
        return True
