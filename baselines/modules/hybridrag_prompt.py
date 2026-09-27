class HybridRAGPromptDesigner:
    """Implement the hybrid ragprompt designer component."""
    def __init__(self):
        pass

    def prompt(self, requirement: str, dll: str, retrieved_docs: list[dict]) -> str:
        """Build the prompt."""
        docs_text = []
        for index, doc in enumerate(retrieved_docs, start=1):
            docs_text.append(
                "\n".join(
                    [
                        f"[Retrieved API {index}]", str(doc.get("api_doc", "")),
                    ]
                )
            )

        retrieved_context = "\n\n".join(docs_text) if docs_text else "No API document was retrieved."

        return f"""
As a developer specializing in deep learning, you are expected to complete the following DL code generation task according to the retrieved API documents.

# [Task Requirement]:
{requirement}

# [Retrieved API Documents]:
{retrieved_context}

# [Task Instructions]:
- Import {dll} and all necessary APIs of {dll} referenced above;
- Prefer APIs that are relevant to the task and consistent with the retrieved API documents.
- Complete the 'DL_Model' class by inheriting the appropriate base model class of {dll};
- Use explicit keyword arguments for all layer API parameters;
- Instantiate 'DL_Model' class and provide a dummy input tensor according to the expected input shape for validation.

# [Code Template]:
```python
class DL_Model:
    # your code here

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = ...
    output_tensor = model(input_tensor)
```
"""