class SecondPromptDesigner:
    """Implement the second prompt designer component."""

    def __init__(self):
        pass

    def prompt(self, task_requirement, api_docs, dll):
        """Build the prompt."""
        api_doc = "".join(api_docs)
        prompt = f"""
As a developer specializing in deep learning, you are expected to complete the following DL code generation task refer to the API document.

# [Task Requirement]:
"{task_requirement}"

# [API Document]:
{api_doc}

# [Task Instructions]:
- Import {dll} and all necessary APIs of {dll} referenced above;
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

        return prompt
