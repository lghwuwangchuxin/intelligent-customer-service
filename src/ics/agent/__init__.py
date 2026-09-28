"""L4 agent：运行态引擎。

🔴 **代码里不存在任何一个具体 Agent**，只有引擎与范式。
   新增 Agent = 插一行配置（agents/*.yaml），不发版。
🔴 编排框架（langchain / langgraph）**只允许出现在本层**（由 .importlinter 强制）。
   换框架时爆炸半径 = engine.py + patterns/。
"""
from .schema import AgentConfig, GuardCfg, RetrievalCfg

__all__ = ["AgentConfig", "GuardCfg", "RetrievalCfg"]
