"""Agent 配置加载。

🔴 加 Agent = 加一个 yaml 文件，不改代码、不新建进程、不加端口。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..core.config import settings
from ..core.errors import AgentNotFound
from ..core.telemetry import get_logger
from .schema import AgentConfig

logger = get_logger(__name__)


class AgentStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = Path(directory or settings.agents_dir)
        self._configs: dict[str, AgentConfig] = {}
        self.reload()

    def reload(self) -> None:
        configs: dict[str, AgentConfig] = {}
        for path in sorted(self._dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text("utf-8")) or {}
            cfg = AgentConfig.model_validate(raw)
            configs[cfg.id] = cfg
        self._configs = configs
        logger.info("已载入 %d 个 Agent 配置（%s）", len(configs), self._dir)

    def get(self, agent_id: str) -> AgentConfig:
        try:
            return self._configs[agent_id]
        except KeyError:
            raise AgentNotFound(
                f"未注册的 Agent: {agent_id}", available=sorted(self._configs)
            ) from None

    def list(self, tenant_id: str | None = None) -> list[AgentConfig]:
        return [
            c for c in self._configs.values()
            if tenant_id is None or c.tenant_id in ("*", tenant_id)
        ]

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._configs


_store: AgentStore | None = None


def get_agent_store() -> AgentStore:
    global _store
    if _store is None:
        _store = AgentStore()
    return _store
