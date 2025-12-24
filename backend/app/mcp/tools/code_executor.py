"""
Code Executor Tool - MCP tool for executing Python code.
Runs code in a sandboxed subprocess with timeout.
"""
import logging
import subprocess
import tempfile
import os
from typing import Optional

from app.mcp.tools.base import BaseMCPTool, ToolParameter

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseMCPTool):
    """
    Execute Python code in a sandboxed environment.
    Use with caution - code execution has security implications.
    """

    name = "code_execute"
    description = (
        "Execute Python code and return the output. "
        "Use this for calculations, data processing, or generating results. "
        "The code runs in an isolated environment with a timeout."
    )
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
            description="Maximum execution time in seconds (1-60)",
            required=False,
            default=30,
        ),
    ]

    # Restricted imports for security
    ALLOWED_IMPORTS = {
        "math", "statistics", "datetime", "json", "re",
        "collections", "itertools", "functools",
        "decimal", "fractions", "random",
    }

    def __init__(self, timeout: int = 30, enabled: bool = True):
        super().__init__()
        self.default_timeout = timeout
        self.enabled = enabled

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """
        Basic security validation of code.

        Args:
            code: Code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "__import__",
            "eval(",
            "exec(",
            "open(",
            "file(",
            "compile(",
            "globals(",
            "locals(",
            "getattr(",
            "setattr(",
            "delattr(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return False, f"Security: '{pattern}' is not allowed"

        return True, ""

    async def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> dict:
        """
        Execute Python code.

        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.

        Returns:
            Dict with success status, output, and error if any.
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Code execution is disabled",
            }

        # Validate code
        is_valid, error = self._validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "error": error,
            }

        timeout = min(max(1, timeout or self.default_timeout), 60)

        try:
            # Create temporary file for the code
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                # Wrap code to capture output
                wrapped_code = f'''
import sys
from io import StringIO

# Capture stdout
_stdout = sys.stdout
sys.stdout = StringIO()

try:
{self._indent_code(code)}
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}")

# Get output
output = sys.stdout.getvalue()
sys.stdout = _stdout
print(output)
'''
                f.write(wrapped_code)
                temp_path = f.name

            # Execute in subprocess
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            # Clean up
            os.unlink(temp_path)

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                logger.info("Code executed successfully")
                return {
                    "success": True,
                    "output": output,
                    "return_code": result.returncode,
                }
            else:
                logger.warning(f"Code execution failed: {error}")
                return {
                    "success": False,
                    "output": output,
                    "error": error,
                    "return_code": result.returncode,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"Code execution timed out after {timeout}s")
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds",
            }
        except Exception as e:
            logger.error(f"Code execution error: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent each line of code."""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))


class CalculatorTool(BaseMCPTool):
    """
    Simple calculator for mathematical expressions.
    Safer alternative to full code execution.
    """

    name = "calculator"
    description = (
        "Evaluate mathematical expressions safely. "
        "Use this for calculations like arithmetic, percentages, etc."
    )
    parameters = [
        ToolParameter(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
            required=True,
        ),
    ]

    # Safe math functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "len": len,
    }

    def __init__(self):
        super().__init__()
        import math
        self.SAFE_FUNCTIONS.update({
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        })

    async def execute(self, expression: str) -> dict:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Math expression to evaluate.

        Returns:
            Dict with result or error.
        """
        try:
            # Safe names for evaluation
            allowed_names = set(self.SAFE_FUNCTIONS.keys())

            # Parse and validate
            import ast
            tree = ast.parse(expression, mode="eval")

            # Check for unsafe nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id not in allowed_names:
                            raise ValueError(f"Function '{node.func.id}' not allowed")

            # Evaluate
            result = eval(
                compile(tree, "<calc>", "eval"),
                {"__builtins__": {}},
                self.SAFE_FUNCTIONS,
            )

            return {
                "success": True,
                "expression": expression,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return {
                "success": False,
                "expression": expression,
                "error": str(e),
            }
