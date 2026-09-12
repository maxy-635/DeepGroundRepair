from typing import Any
from step_two.root_cause_masking.root_cause_preprocessor import ForwardLocalCallSimplifier


class DirectRepairPromptDesigner:
    """Implement the direct repair prompt designer component."""

    def __init__(self, step2_result: dict[str, Any]):
        self.step2_result = step2_result
        self.root_cause = self.step2_result.get("root_cause")
        self.step2_final_code = self.step2_result.get("final_code", "")
        self.forward_call_simplifier = ForwardLocalCallSimplifier()

    def extract_buggy_api_calls(self) -> list[dict[str, Any]]:
        """Extract buggy api calls."""
        init_definition = self.root_cause.get("init_definition") or {}
        if init_definition.get("masked_api_call"):
            return [
                {
                    "buggy_api_call_index": 0,
                    "line_no": init_definition.get("line_no"),
                    "masked_api_call": init_definition.get("masked_api_call"),
                }
            ]

        forward_definition = self.root_cause.get("forward_definition") or {}
        if forward_definition.get("original_call"):
            masked_api_call = self.forward_call_simplifier.simplify(
                forward_definition.get("original_call", "")
            )
            return [
                {
                    "buggy_api_call_index": 0,
                    "line_no": forward_definition.get("line_no"),
                    "masked_api_call": masked_api_call
                    or forward_definition.get("original_call", ""),
                }
            ]

        return []

    def build_buggy_api_calls_block(self) -> str:
        """Build buggy api calls block."""
        buggy_api_calls = sorted(
            self.extract_buggy_api_calls(),
            key=lambda call: call.get("buggy_api_call_index", 0)
        )

        api_call_blocks = []
        for i, api_call in enumerate(buggy_api_calls):
            buggy_api_call_index = api_call.get("buggy_api_call_index", i)
            api_call_blocks.append(
f"""
# [Masked API Call]:

For the buggy API call in the code above, the buggy parameter(s) are masked and represented as follows:
```python
{api_call.get('masked_api_call', '')}
```
"""
)

        return "\n".join(api_call_blocks)


    def prompt(self):
        """Build the prompt."""
        buggy_api_call_blocks = self.build_buggy_api_calls_block()

        prompt = f"""
You are a deep learning code repair assistant. Given tensor-shape-annotated code, and the extracted buggy API calls with buggy parameters masked, your task is to repair the buggy API call.

# [Tensor-Shaped Annotated and Buggy-Parameters-Masked Code]:

```python
{self.step2_final_code}
```

{buggy_api_call_blocks}

# [Task Instructions]:

1. Read the masked API call and identify all masked parameters in the masked API call.
2. Infer valid values for the masked parameters refer to the annotated tensor shapes.
3. Replace only the `'mask'` placeholders in the buggy API call with valid parameter values, ensuring that the resulting API call satisfies all tensor-shape constraints in the given context.
4. Return only the repaired API call as a single Python code line, preserving the masked API call's form.

```python

```
"""

        return prompt
