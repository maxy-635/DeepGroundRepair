class FirstPromptDesigner:
    """Implement the first prompt designer component."""

    def __init__(self):
        pass

    def prompt(self, task_requirement, dll):

        prompt = f"""
There is a deep learning code generation task described below, please provide only the most relevant {dll} APIs for constructing the model.

# [Task Requirement]:
"{task_requirement}"

# [Task Instructions]:
- Return only valid Python code in the following exact format -- a list of dictionaries, each containing the selected API name.

# [Output Format]:
```python
candidate_apis = [{{'api_name':'...'}}, ...]
```
"""
        return prompt