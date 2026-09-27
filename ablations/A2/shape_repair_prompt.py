class ShapeOnlyRepairPromptDesigner:
    """Implement the shape only repair prompt designer component."""

    def __init__(self, step2_result: dict):
        self.step2_result = step2_result
        self.shape_annotated_code = step2_result.get("final_code", "")
        self.traceback_text = (step2_result.get("runtime_error_info") or {}).get("traceback", "")

    def prompt(self) -> str:
        return f"""
You are a deep learning code repair assistant. Given a shape-annotated buggy code and the runtime traceback text, your task is to locate the bug and then return the repaired code. 

# [Shape-Annotated Buggy Code]:
```python
{self.shape_annotated_code}
```

# [Runtime Error Traceback]:
```text
{self.traceback_text}
```

# [Task Instructions]:
1. Read the shape-annotated buggy code and the runtime traceback text carefully.
2. Repair the program with the smallest necessary change that fixes the runtime error.
3. Return only the repaired Python code in one block as follows.
```python

```
"""
