"""L3-c memory：会话与长期记忆。

🔴 与 rag / tools / guard **平级且互不 import**。
🔴 两层必须分清：
   checkpointer —— 本次会话的图状态（消息、工具结果、中断点），按 thread_id 存取
   store        —— 跨会话长期记忆（用户偏好、历史结论），按 namespace 存取
   把长期记忆放进 checkpointer 会让状态无限膨胀；
   把会话状态放进 store 则丢失中断续跑能力。
"""
from .conversation import ConversationIndex, ConversationSummary
from .longterm import LongTermMemory, namespace_for

__all__ = ["ConversationIndex", "ConversationSummary", "LongTermMemory", "namespace_for"]
