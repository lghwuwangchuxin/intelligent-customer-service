"""L1 infra：外部系统的连接与生命周期。

只依赖 core。不含任何业务语义——它只知道"怎么连上 PostgreSQL / Redis"，
不知道"chunk 是什么"。
"""
