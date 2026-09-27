from typing import Any


class NoShapeContextPromptDesigner:
    """Implement the no shape context prompt designer component."""

    def __init__(self, prompt_context: dict[str, Any]):
        self.masked_code = prompt_context.get("masked_code", "")

    def prompt(self) -> str:
        return f"""
You are a deep learning code repair assistant. Given a buggy code with root-cause API parameters masked, your task is to repair the code.

# [Buggy-Parameters-Masked Code]:
```python
{self.masked_code}
```

# [Task Instructions]:
1. Read the buggy code carefully.
2. Repair the program with minimal changes to replace the masked API parameters.
3. Return only the repaired Python code in one block as follows.
```python

```
"""
