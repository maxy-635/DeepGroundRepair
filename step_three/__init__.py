from step_three.handlers import (
    DirectRepairHandler,
    MergeMismatchHandler,
    UnresolvedCrashHandler,
)
from step_three.prompts.direct_repair_prompt import DirectRepairPromptDesigner
from step_three.merge_mismatch_repair.params_solver import (
    MismatchCandidateAdapter,
    MismatchParametersSolver,
)
from step_three.prompts.merge_mismatch_prompt import (
    MergeMismatchPromptDesigner,
)
from step_three.prompts.unresolved_crash_prompt import UnresolvedCrashPromptDesigner
from step_three.result import RepairResult


def __getattr__(name: str):
    """Lazy package exports that avoid importing the executable pipeline on package import."""
    if name == "CodeRegenerationPipeline":
        from step_three.main import CodeRegenerationPipeline

        return CodeRegenerationPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CodeRegenerationPipeline",
    "DirectRepairHandler",
    "MergeMismatchHandler",
    "UnresolvedCrashHandler",
    "DirectRepairPromptDesigner",
    "MergeMismatchPromptDesigner",
    "UnresolvedCrashPromptDesigner",
    "RepairResult",
    "MismatchCandidateAdapter",
    "MismatchParametersSolver",
]
