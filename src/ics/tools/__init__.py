"""L3-b tools：工具与 MCP 客户端。

🔴 与 rag / memory / guard **平级且互不 import**。
🔴 MCP SDK 只允许出现在 ``mcp_client.py``（由 .importlinter 强制）——
   MCP 规范一年内已重构两次，把它关在一个文件里，换协议版本时爆炸半径 = 1 个文件。
"""
