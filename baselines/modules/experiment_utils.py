import os
import json
import shutil
from pathlib import Path


def load_local_llm(model_id: str, model_cache_path: str, data_dtype=None):
    """Load local llm."""

    import torch
    from utils.chat2gemma import Chat2GemmaLLM
    from utils.chat2mistral import Chat2MistralLLM

    if data_dtype is None:
        data_dtype = torch.bfloat16

    normalized_model_id = model_id.lower()

    if "gemma" in normalized_model_id:
        chat2llm = Chat2GemmaLLM
        model, tokenizer = chat2llm().load_model(
            model_id, model_cache_path, data_dtype
        )
    elif "mistral" in normalized_model_id:
        chat2llm = Chat2MistralLLM
        model, tokenizer = chat2llm().load_model(
            model_id, model_cache_path, data_dtype
        )

    else:
        raise ValueError("The paper experiments support only Gemma and Mistral models.")

    return model, tokenizer, chat2llm()


def baseline_id(name: str) -> str:
    """Return the baseline identifier."""
    return name.strip().lower()


def code_dir(llm_name: str, baseline: str, dll: str, experiment_id: str, task_id: str) -> str:
    return f"./baselines/results/{baseline}/response/{llm_name}/{dll}/{task_id}"


def log_dir(llm_name: str, baseline: str, dll: str, experiment_id: str, task_id: str) -> str:
    return f"./baselines/results/{baseline}/chat_logs/{llm_name}/{dll}/{task_id}"


def save_json(data: dict | list, path: str) -> None:
    """Save json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    """Load json."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_text(path: str) -> str:
    """Read text."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def write_text(path: str, text: str) -> None:
    """Write text."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def copy_code_to_result(source_path: str, target_dir: str, count: int, suffix: str = "reused") -> str:
    """Copy code to result."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"code_{suffix}_{count}.py")
    shutil.copyfile(source_path, target_path)
    return target_path


def find_latest_code_by_count(code_path: str, count: int) -> str:
    """Find latest code by count."""
    code_dir_path = Path(code_path)
    if not code_dir_path.exists():
        raise FileNotFoundError(f"Missing code directory: {code_path}")

    candidates = sorted(
        code_dir_path.glob(f"code_*_{count}.py"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Missing generated code for count={count}: {code_path}")
    return str(candidates[0])


def require_file(path: str, purpose: str) -> None:
    """Require file."""
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
