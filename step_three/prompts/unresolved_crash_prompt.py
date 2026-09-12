from typing import Any


class UnresolvedCrashPromptDesigner:
    """
    UnresolvedCrashPromptDesigner: a class to design the unresolved crash prompting.
    """

    def __init__(self, step2_result: dict[str, Any]):
        self.step2_result = step2_result
        self.root_cause = step2_result.get("root_cause")
        self.traceback_text = step2_result.get("runtime_error_info", {}).get("traceback", "")
        self.step2_final_code = step2_result.get("final_code", "")

    def prompt(self) -> str:
        """Build the prompt."""

        prompt = f"""
You are a deep learning code repair assistant. Given a buggy model program and
the runtime traceback, your task is to repair the code.

# [Buggy Code]:

```python
{self.step2_final_code}
```

# [Runtime Error Feedback]:

```text
{self.traceback_text}
```

# [Task Instructions]:

1. Read the buggy code and the runtime traceback carefully.
2. Repair the program with minimal local modifications while preserving the original model structure and design intent.
3. Do not change the model structure, including adding, removing, replacing layers, or introducing new branches. Do not insert extra pooling, convolution, projection, or reshape operations merely to artificially make the code executable.
4. Return only the repaired Python code in one code block as follows.

# [Code Template]:
```python

```
"""

        return prompt
