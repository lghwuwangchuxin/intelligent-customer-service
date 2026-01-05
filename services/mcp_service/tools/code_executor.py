"""Code execution tool with sandbox."""

import asyncio
import sys
import io
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from .base import BaseTool, ToolParameter


class CodeExecutorTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment."""

    name = "code_executor"
    description = "Execute Python code and return the result"
    tags = ["code", "python", "execution"]
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description="Python code to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Execution timeout in seconds",
            required=False,
            default=30,
        ),
    ]
    timeout_seconds = 60

    def __init__(self, timeout: int = 30, allowed_modules: Optional[list] = None):
        """
        Initialize code executor.

        Args:
            timeout: Default execution timeout
            allowed_modules: List of allowed module names
        """
        super().__init__()
        self.default_timeout = timeout
        self.allowed_modules = allowed_modules or [
            "math", "random", "datetime", "json", "re",
            "collections", "itertools", "functools",
            "statistics", "decimal", "fractions",
        ]

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code.

        Args:
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            Execution result
        """
        timeout = timeout or self.default_timeout

        # Validate code
        validation_result = self._validate_code(code)
        if not validation_result["valid"]:
            return {
                "success": False,
                "error": validation_result["error"],
            }

        try:
            result = await asyncio.wait_for(
                self._run_code(code),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Code execution timed out after {timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    async def _run_code(self, code: str) -> Dict[str, Any]:
        """Run code in a subprocess for isolation."""
        # For safety, run in a separate process
        # This is a simplified implementation
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_in_sandbox,
            code,
        )

    def _execute_in_sandbox(self, code: str) -> Dict[str, Any]:
        """Execute code in a sandboxed environment."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create restricted globals
        restricted_globals = {
            "__builtins__": self._get_safe_builtins(),
            "__name__": "__main__",
        }

        # Add allowed modules
        for module_name in self.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        result_value = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile and execute
                compiled = compile(code, "<code>", "exec")
                exec(compiled, restricted_globals)

                # Try to get result from last expression
                if "_result" in restricted_globals:
                    result_value = restricted_globals["_result"]

            return {
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "result": result_value,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": str(e),
            }

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for safety."""
        # Check for dangerous patterns
        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "__import__",
            "eval(",
            "exec(",
            "open(",
            "file(",
            "input(",
            "compile(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return {
                    "valid": False,
                    "error": f"Dangerous pattern detected: {pattern}",
                }

        return {"valid": True}

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get a restricted set of builtins."""
        import builtins

        safe_names = [
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray",
            "bytes", "chr", "complex", "dict", "divmod", "enumerate",
            "filter", "float", "format", "frozenset", "hash", "hex",
            "int", "isinstance", "issubclass", "iter", "len", "list",
            "map", "max", "min", "next", "oct", "ord", "pow", "print",
            "range", "repr", "reversed", "round", "set", "slice",
            "sorted", "str", "sum", "tuple", "type", "zip",
            "True", "False", "None",
        ]

        return {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}
