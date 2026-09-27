class ZeroshotPromptDesigner:
    """Implement the zeroshot prompt designer component."""

    def __init__(self):
        """Initialize the instance."""
        pass

    def prompt(self, requirement, dll):
        """Build the prompt."""

        prompt = f"""
As a developer specializing in deep learning, you are expected to complete the following DL code generation task: 

# [Task Requirement]:
"{requirement}"

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
