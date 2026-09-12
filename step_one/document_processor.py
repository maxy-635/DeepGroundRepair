from pprint import pformat
from utils.utils import read_yaml_data


class DocumentProcessor:
    """Implement the document processor component."""
    def __init__(self):
        pass

    def doc2str(self, file_path):
        """Convert an API document to text fields."""
        yaml_data = read_yaml_data(file_path)

        api_name_str = f"*API Name*: {yaml_data.get('2 api_name', '')}"

        api_description = yaml_data.get("3 api_description", "")
        api_description_str = ""
        if api_description:
            api_description_str += f"*API Description*: {api_description}"

        api_signatures = yaml_data.get('4 api_signature', '')
        api_signatures_str = ""
        if api_signatures:
            api_signatures_str += f"*API Signature*: {api_signatures}"

        api_details = yaml_data.get("5 api_details", "")
        api_details_str = ""
        if api_details:
            api_details_str += f"*API Usage Details*:\n{api_details}"
        else:
            api_details_str += ""
        
        # API Usage Description
        api_usage_description_str = ""
        if api_description or api_details:
            api_usage_description_str += f"*API Usage Description*:\n{api_description} {api_details}"
        api_parameters = yaml_data.get("6 api_parameters", [])
        api_parameters_str = ""
        if api_parameters:
            api_parameters_str += "*API Parameters*:"

            for param in api_parameters: 
                for name, desc in param.items():
                    if name and desc:
                        api_parameters_str += f"\n{[name]}: {desc.strip()}"
        else:
            api_parameters_str += "*API Parameters*:None"

        api_usage = yaml_data.get("7 api_usage_example", [])
        api_usage_str = ""
        if api_usage:
            api_usage_str += "*API Usage Example*:\n```python\n"
            for line in api_usage:
                api_usage_str += line.strip(" '") + "\n"
            api_usage_str += "```"
        else:
            api_usage_str += "*API Usage Example*:None"

        api_shape_formula = yaml_data.get("8 api_shape", "")
        api_shape_formula_str = ""
        if api_shape_formula:
            formatted_api_shape_formula = {}

            input_shape = api_shape_formula.get("input tensor shape")
            if input_shape:
                formatted_api_shape_formula["input tensor shape"] = str(input_shape).strip()
            output_shape = api_shape_formula.get("output tensor shape")
            if output_shape:
                formatted_api_shape_formula["output tensor shape"] = str(output_shape).strip()
            shape_transform_formulas = (
                api_shape_formula.get("tensor shape transform formulas")
                or api_shape_formula.get("formulas")
            )
            if shape_transform_formulas:
                formatted_api_shape_formula["tensor shape transform formulas"] = shape_transform_formulas

            formatted_shape_formula = pformat(formatted_api_shape_formula, width=200, sort_dicts=False)

            api_shape_formula_str += f"*Tensor Shape Transformation Formula*:\n{formatted_shape_formula}"
        else:
            api_shape_formula_str += "*Tensor Shape Transformation Formula*:None"

        return api_name_str, api_description_str, api_signatures_str, api_details_str, api_usage_description_str, api_parameters_str, api_shape_formula_str, api_usage_str
