from typing import Any
from step_three.merge_mismatch_repair.buggy_shape_modeling import BuggyShapeModeler
from step_three.merge_mismatch_repair.retrieve4shapeknowledge import ShapeKnowledgeRetriever



class MergeMismatchPromptDesigner:
    """Implement the merge mismatch prompt designer component."""

    def __init__(self, step2_result: dict[str, Any]):
        self.step2_result = step2_result
        self.root_cause = step2_result.get("root_cause") or {}
        self.masked_annotated_code = step2_result.get("final_code", "")
        self.buggy_api_calls = self.root_cause.get("target_layers") or []
        self.shape_modeler = BuggyShapeModeler(step2_result)
        self.dll = self._infer_dll()
        self.shape_knowledge_retriever = ShapeKnowledgeRetriever(dll=self.dll)

    def _infer_dll(self) -> str:
        """Infer dll."""
        merge_context = self.root_cause.get("merge_context") or {}
        framework = str(merge_context.get("framework") or "").lower()
        framework_to_dll = {
            "tensorflow": "TensorFlow",
            "pytorch": "PyTorch",
            "paddle": "PaddlePaddle",
        }
        if framework not in framework_to_dll:
            raise ValueError(f"Unsupported or missing DL framework: {framework!r}")

        return framework_to_dll[framework]

    def _get_buggy_api_call(self, buggy_api_call_index: Any) -> dict[str, Any]:
        """Return buggy api call."""
        if not isinstance(buggy_api_call_index, int):
            return {}
        if not 0 <= buggy_api_call_index < len(self.buggy_api_calls):
            return {}
        return self.buggy_api_calls[buggy_api_call_index]

    def build_buggy_api_calls_block(self) -> str:
        """Build buggy api calls block."""
        buggy_api_call_shape_pairs = []
        for shape_pair in self.shape_modeler.extract_target_layer_io_shape_pairs():
            buggy_api_call_shape_pairs.append(
                {
                    **shape_pair,
                    "buggy_api_call_index": shape_pair.get("target_layer_index"),
                }
            )
        buggy_api_call_shape_pairs = sorted(
            buggy_api_call_shape_pairs,
            key=lambda shape_pair: (
                shape_pair.get("buggy_api_call_index")
                if isinstance(shape_pair.get("buggy_api_call_index"), int)
                else len(self.buggy_api_calls)
            ),
        )

        api_call_blocks = []
        for i, shape_pair in enumerate(buggy_api_call_shape_pairs):
            buggy_api_call_index = shape_pair.get("buggy_api_call_index", i)
            buggy_api_call = self._get_buggy_api_call(buggy_api_call_index)
            mismatch_types = buggy_api_call.get("mismatch_types") or []
            mismatch_type_text = ", ".join([t.capitalize() for t in mismatch_types]) if mismatch_types else "Unknown"
            retrieved_api_doc = self.shape_knowledge_retriever.main(
                target_layer=buggy_api_call,
                source_code=self.masked_annotated_code,
            )

            api_call_blocks.append(
f"""# For the "Extracted Buggy API Call {buggy_api_call_index}", you should pay attention to "{mismatch_type_text} Dimension" consistency in the following analysis.

## [Shape Constraint Solving Information]:

### Input Tensor Shape: {shape_pair.get('modeled_input_shape')}
### Masked API Call: {buggy_api_call.get('masked_api_call')}
### Target Output Tensor Shape: {shape_pair.get('oracle_output_shape')}
### {retrieved_api_doc.get('tensor_shape_transform_formula')}

## [API Documentation]:

*API Name*: {retrieved_api_doc.get('api_name')}
{retrieved_api_doc.get('api_parameters')}
"""
)

        return "\n".join(api_call_blocks)

    def prompt(self) -> str:
        """Build the prompt."""
        buggy_api_calls_block = self.build_buggy_api_calls_block()

        prompt = f"""
You are a deep learning tensor-shape constraint solving assitant. Your task is to infer valid values for the masked parameters such that, when substituted into the formula, the masked API call alone transforms the input tensor shape exactly into the target output tensor shape.

# [Tensor-Shaped Annotated and Buggy-Parameters-Masked Code]:

```python
{self.masked_annotated_code}
```


{buggy_api_calls_block}

# [Task Instructions]:

1. Identify the masked parameters: 
   Read the masked API call and identify only the parameters whose values are masked.

2. Extract relevant constraints information:
   Read [Tensor-Shaped Annotated and Buggy-Parameters-Masked Code], [Shape Constraint Solving Information], and [API Documentation]. Extract the input tensor shape, target output tensor shape, tensor shape transformation formula, and the meanings of the masked parameters.

3. Generate and verify candidate parameters:
   3.1 Generate candidates: 
       Propose up to 5 candidate combinations for the masked parameters.
   3.2 Verify output shapes: 
       For each valid candidate, substitute its values into the shape transformation formula and compute the output tensor shape.
   3.3 Keep only exact matches:
       Retain only candidates whose computed output shape exactly matches the target output shape.

4. Complete the tasks above step by step. Please provide a reasoning process, then return the final answer as a JSON object following the schema below.
    [  
        {{"buggy_api_call_index": 0,
          "masked_api_call": "<masked API call>",
          "candidate_params_combinations": [
            {{"<masked_param_1>": <value>, "<masked_param_2>": <value>, ...}},
            {{"<masked_param_1>": <value>, "<masked_param_2>": <value>, ...}},
            ...
            ]    
        }}
        ...
    ]
"""

        return prompt
