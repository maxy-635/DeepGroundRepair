from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List

try:
    import torch
except ImportError:
    torch = None

try:
    import paddle
except ImportError:
    paddle = None


class SequentialRuntimeFailureTracker:
    """Implement the sequential runtime failure tracker component."""

    def __init__(self):
        """Initialize the instance."""
        self.container_failures: List[Dict[str, object]] = []

    def reset(self):
        self.container_failures = []

    @contextmanager
    def track(self):
        self.reset()
        patches = self.build_patches()
        self.apply_patches(patches)
        try:
            yield self
        finally:
            self.restore_patches(patches)

    def dumps_failures(self):
        """Return failures."""
        return [dict(item) for item in self.container_failures]

    def build_patches(self):
        """Build patches."""
        patches = []

        torch_nn = getattr(torch, "nn", None) if torch is not None else None
        torch_sequential = getattr(torch_nn, "Sequential", None)
        if torch_sequential is not None and hasattr(torch_sequential, "forward"):
            patches.append(
                self.build_patch_record(
                    sequential_cls=torch_sequential,
                    method_name="forward",
                    framework_name="pytorch",
                )
            )

        paddle_nn = getattr(paddle, "nn", None) if paddle is not None else None
        paddle_sequential = getattr(paddle_nn, "Sequential", None)
        if paddle_sequential is not None and hasattr(paddle_sequential, "forward"):
            patches.append(
                self.build_patch_record(
                    sequential_cls=paddle_sequential,
                    method_name="forward",
                    framework_name="paddle",
                )
            )

        return patches

    def build_patch_record(self, sequential_cls, method_name: str, framework_name: str):
        """Build patch record."""
        original_method = getattr(sequential_cls, method_name)
        patched_method = self.make_wrapped_sequential_method(framework_name)
        return {
            "sequential_cls": sequential_cls,
            "method_name": method_name,
            "original_method": original_method,
            "patched_method": patched_method,
        }

    def apply_patches(self, patches):
        """Apply patches."""
        for patch in patches:
            setattr(
                patch["sequential_cls"],
                patch["method_name"],
                patch["patched_method"],
            )

    def restore_patches(self, patches):
        """Restore patches."""
        for patch in reversed(patches):
            setattr(
                patch["sequential_cls"],
                patch["method_name"],
                patch["original_method"],
            )

    def make_wrapped_sequential_method(self, framework_name: str):
        """Create wrapped sequential method."""

        tracker = self

        def wrapped_forward(sequential_instance, input_tensor):
            modules = tracker.list_child_modules(sequential_instance)
            current_value = input_tensor

            for child_index, child_module in enumerate(modules):
                try:
                    current_value = child_module(current_value)
                except Exception:
                    tracker.record_failure(
                        framework_name=framework_name,
                        sequential_instance=sequential_instance,
                        modules=modules,
                        child_index=child_index,
                        child_module=child_module,
                    )
                    raise

            return current_value

        return wrapped_forward

    def list_child_modules(self, sequential_instance):
        """List child modules."""
        modules_dict = getattr(sequential_instance, "_modules", None)
        if isinstance(modules_dict, dict):
            return list(modules_dict.values())

        sub_layers = getattr(sequential_instance, "_sub_layers", None)
        if isinstance(sub_layers, dict):
            return list(sub_layers.values())

        children = getattr(sequential_instance, "children", None)
        if callable(children):
            return list(children())

        return []

    def record_failure(
        self,
        framework_name: str,
        sequential_instance,
        modules,
        child_index: int,
        child_module,
    ):
        """Record failure."""
        child_types = [type(module).__name__ for module in modules]
        failure_item = {
            "framework": framework_name,
            "container_type": type(sequential_instance).__name__,
            "container_size": len(modules),
            "container_child_types": child_types,
            "failed_child_index": child_index,
            "failed_child_type": type(child_module).__name__,
        }
        self.container_failures.append(failure_item)
