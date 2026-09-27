from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from baselines.modules.no_weight_validation import NoWeightRunner
from step_three.main import CodeRegenerationPipeline
from step_three.llm_runtime import LLMRuntime
from step_two.tensor_shape_debug import TensorShapeDebugPipeline
from utils.utils import get_all_files, get_llm_name



def normalize_dll(dll: str) -> str:
    """Normalize dll."""
    normalized = dll.strip().lower()
    return normalized


def model_name(model_id: str) -> str:
    return get_llm_name(model_id)



def ablation_root(ablation_id: str) -> Path:
    return PROJECT_ROOT / "ablations" / "results" / ablation_id


def code_dir(ablation_id: str, stage: str, llm_name: str, dll: str, experiment_id: str, task_id: str | None = None) -> Path:
    path = (
        ablation_root(ablation_id)
        / stage
        / llm_name
        / normalize_dll(dll)
    )
    return path / task_id if task_id else path


def log_dir(ablation_id: str, stage: str, llm_name: str, dll: str, experiment_id: str, task_id: str | None = None) -> Path:
    path = (
        ablation_root(ablation_id)
        / stage
        / "chat_logs"
        / llm_name
        / normalize_dll(dll)
        / f"logs_{experiment_id}"
    )
    return path / task_id if task_id else path


def step2_dir(ablation_id: str, llm_name: str, dll: str, experiment_id: str) -> Path:
    return (
        ablation_root(ablation_id)
        / "step_two"
        / llm_name
        / normalize_dll(dll)
    )


def repair_dir(ablation_id: str, llm_name: str, dll: str, experiment_id: str) -> Path:
    return (
        ablation_root(ablation_id)
        / "step_three"
        / llm_name
        / normalize_dll(dll)
    )


def configure_logger(ablation_id: str, model_id: str, experiment_id: str, stage: str, dll: str) -> int:
    """Configure logger."""
    llm_name = get_llm_name(model_id)
    path = PROJECT_ROOT / "ablations" / "logs" / ablation_id / stage / llm_name / normalize_dll(dll)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / f"{experiment_id}.log"
    return logger.add(str(log_file))


def configure_sampling_config_sinks(
    ablation_id: str,
    stage: str,
    model_id: str,
    experiment_id: str,
    dlls: list[str],
) -> list[int]:
    """Temporarily route one sampling-configuration event to every DLL log file."""
    llm_name = get_llm_name(model_id)
    sink_ids = []
    for dll in dict.fromkeys(normalize_dll(dll) for dll in dlls):
        path = PROJECT_ROOT / "ablations" / "logs" / ablation_id / stage / llm_name / dll
        path.mkdir(parents=True, exist_ok=True)
        log_file = path / f"{experiment_id}.log"
        sink_ids.append(
            logger.add(
                str(log_file),
                filter=lambda record: record["extra"].get("sampling_config", False),
            )
        )
    return sink_ids


def remove_log_sinks(sink_ids: list[int]) -> None:
    """Remove temporary Loguru sinks created for a batch-level configuration event."""
    for sink_id in sink_ids:
        logger.remove(sink_id)


def build_llm_runtime(model_id: str, cache_dir: str | None, max_tokens: int) -> LLMRuntime:
    """Build llm runtime."""
    return LLMRuntime.hf(model_id=model_id, cache_dir=cache_dir, max_tokens=max_tokens)


def iter_benchmark_tasks(benchmark_path: str) -> list[str]:
    return get_all_files(str(PROJECT_ROOT / benchmark_path) if not os.path.isabs(benchmark_path) else benchmark_path, ".yaml")


def task_id_from_yaml(yaml_file: str) -> str:
    return Path(yaml_file).stem


def iter_pyfiles(code_root: str | Path) -> list[Path]:
    root = Path(code_root)
    if not root.exists():
        return []
    return [Path(path) for path in get_all_files(str(root), ".py")]


def has_code_for_count(task_dir: str | Path, count: int) -> bool:
    path = Path(task_dir)
    return path.exists() and any(path.glob(f"code_*_{count}.py"))


def safe_relative_to(path: str | Path, root: str | Path) -> Path:
    return Path(path).resolve().relative_to(Path(root).resolve())


def annotated_path_for_pyfile(pyfile: Path, source_root: Path, target_root: Path) -> Path:
    relative = safe_relative_to(pyfile, source_root)
    return target_root / relative.with_name(f"{relative.stem}_annotated.py")


def patch_no_weight_for_ablation_step2(dll_type: str) -> None:
    """Patch no weight for ablation step2."""
    if dll_type == "tensorflow":
        NoWeightRunner.patch_tensorflow_keras()
    elif dll_type == "pytorch":
        NoWeightRunner.patch_pytorch()
    elif dll_type == "paddlepaddle":
        NoWeightRunner.patch_paddlepaddle()


def run_step2_on_code_dir(source_code_root: str | Path, target_step2_root: str | Path, dll: str, skip_existing: bool = True) -> list[Path]:
    """Run step2 on code dir."""
    source_root = Path(source_code_root)
    target_root = Path(target_step2_root)
    dll_type = normalize_dll(dll)
    saved_json_files: list[Path] = []

    for pyfile in iter_pyfiles(source_root):
        annotated_pyfile = annotated_path_for_pyfile(pyfile, source_root, target_root)
        result_json = annotated_pyfile.with_suffix(".json")
        if skip_existing and result_json.exists():
            logger.info(f"[Step2] Skipping existing result: {result_json}")
            saved_json_files.append(result_json)
            continue

        logger.info(f"[Step2] Processing: {pyfile}")
        patch_no_weight_for_ablation_step2(dll_type)
        pipeline = TensorShapeDebugPipeline(
            origin_pyfile=str(pyfile),
            annotated_pyfile=str(annotated_pyfile),
            dll_type=dll_type,
        )
        result = pipeline.main()
        if result is None:
            logger.error(f"[Step2] Processing failed: {pyfile}")
            continue
        saved_json_files.append(result_json)

    return saved_json_files


class SafeCodeRegenerationPipeline(CodeRegenerationPipeline):
    """Implement the safe code regeneration pipeline component."""

    def build_repair_code_path(self, step2_result_file: str) -> Path:
        if not self.repair_code_save_path:
            return super().build_repair_code_path(step2_result_file)

        relative = safe_relative_to(step2_result_file, self.step2_result_path)
        repaired_name = relative.name.replace("_annotated.json", "_repaired.py")
        if repaired_name == relative.name:
            repaired_name = relative.with_suffix(".py").name
        return Path(self.repair_code_save_path) / relative.parent / repaired_name


def run_full_step3(
    model_id: str,
    source_step2_root: str | Path,
    target_repair_root: str | Path,
    llm_runtime: LLMRuntime,
) -> None:
    """Run full step3."""
    pipeline = SafeCodeRegenerationPipeline(
        step2_result_path=str(source_step2_root),
        repair_code_save_path=str(target_repair_root),
        llm_runtime=llm_runtime,
    )
    pipeline.main()



def extract_python_code(llm_response: Any) -> str | None:
    """Extract python code."""
    if llm_response is None:
        return None
    text = str(llm_response).strip()
    import re

    match = re.search(r"```python\s*(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    if match:
        return match.group(1).strip()
    if "import " in text or "class " in text or "nn." in text:
        return text
    return None


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save json."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load json."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
