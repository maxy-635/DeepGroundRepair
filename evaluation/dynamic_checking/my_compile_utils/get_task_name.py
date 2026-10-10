import os
import re


def get_task_name(path):
    """Return task name."""
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    entries = os.listdir(path)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    subfolders.sort(key=natural_sort_key)
    
    return subfolders
