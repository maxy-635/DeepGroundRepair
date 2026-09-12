import os
import importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def api_validation(api_full_name: str):
    """Validate that an API is available."""
    api_full_name = api_full_name.strip()

    if "(" in api_full_name:
        api_full_name = api_full_name.split("(", 1)[0]

    parts = api_full_name.split(".")
    if len(parts) < 2:
        return False

    try:
        for i in range(1, len(parts)):
            module_path = ".".join(parts[:i])
            importlib.import_module(module_path)

        parent_module_path = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent_module = importlib.import_module(parent_module_path)
        getattr(parent_module, attr_name)

        return True

    except Exception as e:
        return False
