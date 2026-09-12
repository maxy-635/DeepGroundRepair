# Step Three Handlers

This package dispatches Step Three repair work by root-cause type.

- `base.py`: shared handler utilities.
- `direct_repair_handler.py`: direct repair for diagnosed failures.
- `merge_mismatch_handler.py`: tensor-merge mismatch repair.
- `unresolved_handler.py`: fallback repair for unresolved crashes.

Handlers are invoked by `step_three/main.py`; they are not standalone command-line programs.
