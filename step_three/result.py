from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepairResult:
    """Implement the repair result component."""

    root_cause_type: str
    repaired_code: str = ""
    prompt: str | None = None
    llm_response: Any = None
    repair_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "root_cause_type": self.root_cause_type,
            "repaired_code": self.repaired_code,
            "prompt": self.prompt,
            "llm_response": self.llm_response,
            "repair_meta": self.repair_meta,
        }
