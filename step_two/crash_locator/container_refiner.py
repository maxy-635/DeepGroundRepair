class SequentialContainerInitDefinitionRefiner:
    """Implement the sequential container init definition refiner component."""

    def refine_container_init_definition(self, init_definition, traceback_text: str, runtime_error_info=None):
        """Refine container init definition."""
        children = getattr(init_definition, "children", None) or []
        if not children:
            return init_definition

        target_child = self.select_container_child(
            children,
            traceback_text,
            runtime_error_info=runtime_error_info,
        )
        if target_child is None:
            return init_definition

        from step_two.crash_locator.root_cause_models import InitModuleDefinition

        return InitModuleDefinition(
            module_name=target_child["layer_name"],
            line_no=target_child["init_line_no"],
            code_line=target_child["original_call"],
            module_type=target_child["api_name"],
            end_line_no=target_child.get("end_line_no"),
        )

    def select_container_child(self, children, traceback_text: str, runtime_error_info=None):
        """Select container child."""
        target_child = self.select_container_child_by_runtime_failure(
            children,
            runtime_error_info=runtime_error_info,
        )
        if target_child is not None:
            return target_child

        return self.select_container_child_by_traceback(children, traceback_text)

    def select_container_child_by_runtime_failure(self, children, runtime_error_info=None):
        """Select container child by runtime failure."""
        if not runtime_error_info:
            return None

        container_failures = runtime_error_info.get("container_failures") or []
        if not container_failures:
            return None

        for failure in reversed(container_failures):
            if not self.is_compatible_runtime_failure(failure, children):
                continue

            failed_child_index = failure.get("failed_child_index")
            if not isinstance(failed_child_index, int):
                continue
            if failed_child_index < 0 or failed_child_index >= len(children):
                continue
            return children[failed_child_index]

        return None

    def is_compatible_runtime_failure(self, failure, children):
        if str(failure.get("container_type", "")).lower() != "sequential":
            return False

        if failure.get("container_size") != len(children):
            return False

        runtime_child_types = failure.get("container_child_types") or []
        if not runtime_child_types:
            return True

        expected_child_types = [
            str(child.get("api_name", "")).lower()
            for child in children
        ]
        normalized_runtime_child_types = [
            str(child_type).lower()
            for child_type in runtime_child_types
        ]
        return normalized_runtime_child_types == expected_child_types

    def select_container_child_by_traceback(self, children, traceback_text: str):
        """Select container child by traceback."""
        lowered_traceback = traceback_text.lower()
        best_child = None
        best_score = 0

        for child in children:
            score = self.score_child_against_traceback(child, lowered_traceback)
            if score > best_score:
                best_child = child
                best_score = score
                continue

            if score == best_score and score > 0 and best_child is not None:
                if child.get("child_index", -1) > best_child.get("child_index", -1):
                    best_child = child

        if best_score < 2:
            return None
        return best_child

    def score_child_against_traceback(self, child, lowered_traceback: str):
        """Score child against traceback."""
        api_name = child.get("api_name")
        if not api_name:
            return 0

        api_lower = api_name.lower()
        score = 0

        if f"operator < {api_lower} >" in lowered_traceback:
            score += 3
        if f"f.{api_lower}" in lowered_traceback:
            score += 2
        if f"/{api_lower}.py" in lowered_traceback or f"\\{api_lower}.py" in lowered_traceback:
            score += 2
        if f"in {api_lower}" in lowered_traceback:
            score += 1

        return score
