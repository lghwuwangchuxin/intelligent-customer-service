"""提示词加载。

🔴 提示词放 ``prompts/`` 目录，**不进代码**：
   它是会频繁改动、改坏了很难察觉的东西（表现不是报错，是答案质量慢慢变差）。
   文件名即版本：``router.v2.md``，回滚 = 改 prompt_ref 指针。
"""
from __future__ import annotations

import re
from pathlib import Path

from ..core.config import settings
from ..core.errors import AppError
from ..core.telemetry import get_logger

logger = get_logger(__name__)

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptNotFound(AppError):
    code, http_status = "INTERNAL", 500


class PromptLibrary:
    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = Path(directory or settings.prompts_dir)
        self._cache: dict[str, str] = {}

    def load(self, ref: str) -> str:
        """ref 形如 ``router.v2`` → ``prompts/router.v2.md``。"""
        if ref in self._cache:
            return self._cache[ref]
        path = self._dir / f"{ref}.md"
        if not path.exists():
            raise PromptNotFound(
                f"提示词不存在: {path}",
                available=sorted(p.stem for p in self._dir.glob("*.md")),
            )
        text = path.read_text("utf-8")
        self._cache[ref] = text
        return text

    def render(self, ref: str, **values: object) -> str:
        template = self.load(ref)
        needed = set(_PLACEHOLDER.findall(template))
        missing = needed - set(values)
        if missing:
            # 🔴 缺变量必须报错。静默留空会在提示词里开洞，模型行为变了却查不出原因。
            raise PromptNotFound(f"提示词 {ref} 缺少变量: {sorted(missing)}", ref=ref)
        return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)

    def refs(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.md"))


_library: PromptLibrary | None = None


def get_prompts() -> PromptLibrary:
    global _library
    if _library is None:
        _library = PromptLibrary()
    return _library
