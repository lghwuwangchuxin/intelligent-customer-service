"""架构守卫。

价值不在"现在是对的"，而在**变坏时立刻变红**。
分层违规与死配置都是悄悄长出来的，人工 review 抓不住。
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ics"


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_import_linter_contracts_hold():
    """🔴 四条契约：分层单向 / 能力层互不依赖 / 编排框架关在 agent / MCP SDK 关在客户端。

    这是把「爆炸半径表」从承诺变成事实的机制。
    """
    r = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"import-linter 契约被破坏：\n{r.stdout}\n{r.stderr}"


def test_mcp_sdk_confined_to_two_modules():
    """可 grep 自查的爆炸半径：换 MCP 协议版本只需改这两个文件。"""
    offenders = []
    for p in _modules():
        rel = str(p.relative_to(SRC))
        if rel in {"tools/mcp_client.py", "mcp_servers/knowledge_server.py"}:
            continue
        for node in ast.walk(ast.parse(p.read_text("utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "mcp":
                offenders.append(rel)
            elif isinstance(node, ast.Import):
                offenders += [rel for a in node.names if a.name.split(".")[0] == "mcp"]
    assert not offenders, f"MCP SDK 泄漏到: {sorted(set(offenders))}"


def test_orchestration_framework_confined_to_agent_layer():
    """换编排框架时爆炸半径 = agent/ 目录。"""
    offenders = []
    for p in _modules():
        rel = str(p.relative_to(SRC))
        if rel.startswith("agent/") or rel.startswith("api/"):
            continue
        for node in ast.walk(ast.parse(p.read_text("utf-8"))):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            if any(m.split(".")[0] in {"langchain", "langgraph"} for m in mods):
                offenders.append(rel)
    assert not offenders, f"编排框架泄漏到: {sorted(set(offenders))}"


def test_no_contextvar_for_request_state():
    """🔴 禁用 ContextVar 传请求状态：异步/线程池边界会静默丢失 → 跨租户泄漏。"""
    offenders = []
    for p in _modules():
        tree = ast.parse(p.read_text("utf-8"))
        for node in ast.walk(tree):
            # 只看真实引用，注释里写"禁用 ContextVar"不算违规
            if (isinstance(node, ast.Name) and node.id == "ContextVar") or (isinstance(node, ast.Attribute) and node.attr == "ContextVar") or (isinstance(node, ast.ImportFrom) and node.module == "contextvars"):
                offenders.append(str(p.relative_to(SRC)))
    assert not offenders, f"使用了 ContextVar 传请求状态: {sorted(set(offenders))}"


def test_no_dead_settings():
    """配置声明了就必须有人消费，否则是骗人的开关。"""
    cfg = (SRC / "core" / "config.py").read_text("utf-8")
    others = "\n".join(p.read_text("utf-8") for p in _modules() if p.name != "config.py")
    others += "\n".join(p.read_text("utf-8") for p in (ROOT / "tests").glob("*.py"))

    dead = []
    for node in ast.walk(ast.parse(cfg)):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    if not re.search(rf"\.{re.escape(name)}\b", others):
                        dead.append(f"{node.name}.{name}")
    assert not dead, "配置无人消费（要么接线，要么删掉）：\n  " + "\n  ".join(dead)


def test_no_unused_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    # 运行时/动态加载，不直接 import——逐个写明理由
    RUNTIME_ONLY = {
        "python-multipart",              # FastAPI 解析 multipart 上传
        "python-dotenv",                 # pydantic-settings 读 .env
        "uvicorn",                       # 经 cli.py 字符串路径启动
        "langchain-openai",              # init_chat_model 按 provider 动态加载
        "langchain-ollama",              # 同上（本地默认 provider）
        "langgraph-checkpoint-postgres",  # 动态导入
        "psycopg",                       # 以 psycopg_pool / pgvector.psycopg 使用
    }
    IMPORT_NAME = {
        "pyyaml": "yaml", "beautifulsoup4": "bs4", "python-docx": "docx",
        "pydantic-settings": "pydantic_settings", "prometheus-client": "prometheus_client",
        "opentelemetry-api": "opentelemetry",
        "opentelemetry-sdk": "opentelemetry",
        "opentelemetry-exporter-otlp-proto-http": "opentelemetry",
    }
    corpus = "\n".join(
        p.read_text("utf-8") for p in (*_modules(), *(ROOT / "tests").glob("*.py"))
    )
    unused = []
    for spec in pyproject["project"]["dependencies"]:
        name = re.split(r"[><=\[;]", spec)[0].strip()
        if name in RUNTIME_ONLY:
            continue
        root = IMPORT_NAME.get(name, name.replace("-", "_")).split(".")[0]
        if not re.search(rf"\b(import|from)\s+{re.escape(root)}\b", corpus):
            unused.append(name)
    assert not unused, "依赖从未被 import：\n  " + "\n  ".join(unused)


def test_every_layer_documents_its_boundary():
    """每层 __init__ 写清"什么该放这里"——给未来的人看的。"""
    for pkg in ("core", "infra", "model", "rag", "tools", "memory", "guard",
                "agent", "api", "worker", "mcp_servers"):
        init = SRC / pkg / "__init__.py"
        assert init.exists(), f"{pkg} 缺 __init__.py"
        doc = ast.get_docstring(ast.parse(init.read_text("utf-8")))
        assert doc and len(doc) >= 20, f"{pkg}/__init__.py 缺少职责边界说明"


def test_prompts_are_files_not_code():
    """🔴 提示词放 prompts/，不进代码：它改坏了表现不是报错，是质量慢慢变差。"""
    assert list((ROOT / "prompts").glob("*.md")), "prompts/ 下没有提示词文件"
    # 代码里不应出现长段中文提示词（边界规则等系统约束除外）
    for p in _modules():
        if p.name == "assemble.py":       # BOUNDARY_RULE 是安全约束，必须随代码走
            continue
        text = p.read_text("utf-8")
        for m in re.finditer(r'"""(.{200,}?)"""', text, re.DOTALL):
            body = m.group(1)
            if body.count("你是") or body.count("请回答"):
                raise AssertionError(f"{p.relative_to(SRC)} 内联了提示词，应移到 prompts/")


def test_agents_are_data_not_code():
    """加 Agent 不该需要改代码。"""
    assert list((ROOT / "agents").glob("*.yaml"))
    for p in _modules():
        text = p.read_text("utf-8")
        assert "class GeneralAgent" not in text
        assert "class BillingAgent" not in text
