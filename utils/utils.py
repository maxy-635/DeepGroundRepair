import os
import re
import yaml
import json
from natsort import natsorted



def get_sampling_temperature(model_id: str) -> float:
    """Return sampling temperature."""

    MISTRAL_3_EXPERIMENT_MODEL_IDS = {
        "mistralai/ministral-3-3b-instruct-2512",
        "mistralai/ministral-3-14b-instruct-2512",
        "mistralai/mistral-small-3.2-24b-instruct-2506",
    }

    if model_id.strip().casefold() in MISTRAL_3_EXPERIMENT_MODEL_IDS:
        return 0.1
    return 0.8

def read_yaml_data(yaml_file):
    '''Read data from a YAML file.
    Args:
        yaml_file (str): Path to the YAML file.
    Returns:
        dict: Data read from the YAML file.
    '''
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    data = yaml.load(file_data,Loader=yaml.FullLoader)#dict

    return data


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


def get_llm_name(input):
    """Return llm name."""
    output = re.sub(r"[^a-z0-9]+", "_", input.split("/")[-1].lower()).strip("_")

    return output



def dict2json(source_dict, save_path):
    """Write a dictionary as JSON."""
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(source_dict, json_file, indent=2)


def get_task_name(path)->list:
    """Return task name."""
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    entries = os.listdir(path)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    subfolders.sort(key=natural_sort_key)
    return subfolders
