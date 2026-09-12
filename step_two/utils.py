import os
import re
from natsort import natsorted

def get_llm_name(input):
    """Return llm name."""
    output = re.sub(r"[^a-z0-9]+", "_", input.split("/")[-1].lower()).strip("_")

    return output

def remove_local_path_info(traceback_text):
    """Remove local path info."""
    if not traceback_text:
        return traceback_text

    home_dir = re.escape(os.path.expanduser("~"))
    project_root = re.escape(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    cleaned_text = str(traceback_text)

    cleaned_text = re.sub(project_root + r"/\.?/?", "./", cleaned_text)
    wrapper_frame_pattern = (
        r'\n?\s*File "\.?/?step_two/tensor_shape_trace/tensor_shape_tracer\.py", '
        r'line \d+, in execute_and_trace\n'
        r'\s*exec\(compile\(self\.python_code, self\.filename, "exec"\), '
        r'namespace, namespace\)\n?'
    )
    cleaned_text = re.sub(wrapper_frame_pattern, "\n", cleaned_text)
    cleaned_text = re.sub(r"Traceback \(most recent call last\):\n+", "Traceback (most recent call last):\n", cleaned_text)
    env_lib_pattern = (
        home_dir
        + r"/(?:anaconda3|miniconda3|miniforge3|mambaforge|micromamba|\.conda)"
        + r"(?:/envs/[^/]+)?/lib"
    )
    cleaned_text = re.sub(env_lib_pattern, "", cleaned_text)
    cleaned_text = re.sub(home_dir + r"/", "", cleaned_text)
    cleaned_text = cleaned_text.replace("././", "./")

    return cleaned_text

def find_parameter_end(line_sources, start_lineno, start_index):
    """Find parameter end."""
    stack = []
    single_quote = False
    double_quote = False
    for i, _line in enumerate(line_sources[start_lineno - 1:], start=start_lineno - 1):
        if i != start_lineno - 1:
            start_index = 0
        for j, c in enumerate(line_sources[i][start_index:], start=start_index):
            if c == "'":
                if not double_quote:
                    single_quote = not single_quote
            elif c == '"':
                if not single_quote:
                    double_quote = not double_quote
            elif not single_quote and not double_quote:
                if c in ["(", "[", "{"]:
                    stack.append(c)
                elif c in ["]", "}"]:
                    if stack:
                        stack.pop()
                elif c == ")":
                    if len(stack) == 0:
                        return i + 1, j
                    stack.pop()
                elif c == ",":
                    if len(stack) == 0:
                        return i + 1, j
    return None


def get_all_files(directory, file_type):
    """Return all files."""
    def _get_files_in_directory(current_directory):
        files = os.listdir(current_directory)
        required_files = []

        for file in files:
            full_path = os.path.join(current_directory, file)
            if os.path.isfile(full_path):
                if full_path.endswith(file_type):
                    required_files.append(full_path)
            elif os.path.isdir(full_path):
                required_files.extend(_get_files_in_directory(full_path))

        return required_files

    all_files = _get_files_in_directory(directory)

    sorted_files = natsorted(all_files)

    return sorted_files


def compact_step2_result_for_save(result):
    runtime_error_info = result.get("runtime_error_info")
    compact_runtime_error_info = None
    if runtime_error_info is not None:
        compact_runtime_error_info = {
            "traceback": runtime_error_info.get("traceback", ""),
        }

    return {
        "runtime_error_info": compact_runtime_error_info,
        "root_cause": result.get("root_cause") or {"root_cause_type": "No_Runtime_Error"},
        "origin_code": result.get("origin_code", ""),
        "final_code": result.get("final_code", ""),
    }
