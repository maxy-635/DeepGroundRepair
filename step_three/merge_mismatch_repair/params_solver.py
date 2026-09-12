from __future__ import annotations
import ast, json, re
from typing import Any
from loguru import logger

from step_three.merge_mismatch_repair.buggy_shape_modeling import BuggyShapeModeler
from step_three.merge_mismatch_repair.shape_simulator import ShapeSimulator
from step_three.merge_mismatch_repair.helper.params_solver_utils import (
    extract_call_params,
    find_shape_pair,
    resolve_api_name,
    resolve_merge_ignored_axes,
    assert_same_shape,
)


class MismatchCandidateAdapter:
    """Implement the mismatch candidate adapter component."""

    def get_buggy_api_call_index(self, candidate: dict[str, Any]) -> Any:
        """Return buggy api call index."""
        return candidate.get("buggy_api_call_index")

    def build_candidate_space(
        self,
        llm_raw_response: Any,
        max_candidates: int = 10,
    ) -> list[dict[str, Any]]:
        """Build candidate space."""
        return self.enumerate_candidate_combinations(
            self.extract_candidate_space(llm_raw_response),
            max_candidates=max_candidates,
        )

    def extract_candidate_space(self, llm_raw_response: Any) -> list[dict[str, Any]]:
        """Extract candidate space."""
        if isinstance(llm_raw_response, list):
            return [
                dict(candidate)
                for candidate in llm_raw_response
                if isinstance(candidate, dict)
            ]

        if isinstance(llm_raw_response, dict):
            if isinstance(llm_raw_response.get("candidate_params_combinations"), list):
                return [dict(llm_raw_response)]

            return []

        if not isinstance(llm_raw_response, str):
            return []

        raw_text = llm_raw_response.strip()
        fenced_blocks = re.findall(r"```(?:json|python)?\s*(.*?)```", raw_text, re.DOTALL)
        candidate_texts = [raw_text] + [block.strip() for block in fenced_blocks]

        for candidate_text in candidate_texts:
            if not candidate_text:
                continue

            parsed_candidate = None
            try:
                parsed_candidate = json.loads(candidate_text)
            except json.JSONDecodeError:
                try:
                    parsed_candidate = ast.literal_eval(candidate_text)
                except (ValueError, SyntaxError):
                    continue

            if isinstance(parsed_candidate, list):
                return [
                    dict(candidate)
                    for candidate in parsed_candidate
                    if isinstance(candidate, dict)
                ]

            if not isinstance(parsed_candidate, dict):
                continue

            if isinstance(parsed_candidate.get("candidate_params_combinations"), list):
                return [dict(parsed_candidate)]

        return []

    def get_masked_param_names(self, masked_api_call: str) -> set[str]:
        """Return masked param names."""
        return set(
            re.findall(r"\b([A-Za-z_]\w*)\s*=\s*['\"]?mask['\"]?", masked_api_call)
        )

    def normalize_candidate_literals(self, value: Any) -> Any:
        """Normalize candidate literals."""
        if isinstance(value, dict):
            return {
                key: self.normalize_candidate_literals(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self.normalize_candidate_literals(item)
                for item in value
            ]

        if not isinstance(value, str):
            return value

        text = value.strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value

    def enumerate_candidate_combinations(
        self,
        candidate_space: list[dict[str, Any]],
        max_candidates: int = 10,
    ) -> list[dict[str, Any]]:
        """Enumerate candidate combinations."""
        if not isinstance(candidate_space, list):
            return []

        if all(
            isinstance(item, dict) and isinstance(item.get("buggy_api_calls"), list)
            for item in candidate_space
        ):
            return [
                self.normalize_candidate_literals(candidate)
                for candidate in candidate_space[:max_candidates]
            ]

        candidates = [{"buggy_api_calls": []}]
        for item in candidate_space:
            if not isinstance(item, dict):
                return []

            param_combinations = item.get("candidate_params_combinations")
            if not isinstance(param_combinations, list):
                return []

            next_candidates = []
            for candidate in candidates:
                for params in param_combinations:
                    if not isinstance(params, dict):
                        continue

                    buggy_api_call_candidate = {
                        "buggy_api_call_index": self.get_buggy_api_call_index(item),
                        "masked_api_call": item.get("masked_api_call"),
                        "params": self.normalize_candidate_literals(params),
                    }
                    next_candidates.append(
                        {
                            "buggy_api_calls": candidate["buggy_api_calls"]
                            + [buggy_api_call_candidate]
                        }
                    )
                    if len(next_candidates) >= max_candidates:
                        break
                if len(next_candidates) >= max_candidates:
                    break

            candidates = next_candidates
            if not candidates or len(candidates) >= max_candidates:
                break

        return candidates[:max_candidates]

    def get_buggy_api_calls(self, step2_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return buggy api calls."""
        root_cause = step2_result.get("root_cause") or {}
        buggy_api_calls = root_cause.get("target_layers") or []
        return [
            {**buggy_api_call, "_buggy_api_call_index": index}
            for index, buggy_api_call in enumerate(buggy_api_calls)
            if isinstance(buggy_api_call, dict)
        ]

    def extract_candidate_for_buggy_api_call(
        self,
        candidate: dict[str, Any],
        buggy_api_call: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Extract candidate for buggy api call."""
        target_index = buggy_api_call.get("_buggy_api_call_index")
        target_masked_call = buggy_api_call.get("masked_api_call")
        api_call_items = candidate.get("buggy_api_calls")

        if isinstance(api_call_items, list):
            for item in api_call_items:
                if not isinstance(item, dict):
                    continue

                same_index = self.get_buggy_api_call_index(item) == target_index
                same_masked_call = (
                    target_masked_call
                    and item.get("masked_api_call") == target_masked_call
                )
                if not (same_index or same_masked_call):
                    continue

                params = item.get("params")
                return (dict(params), True) if isinstance(params, dict) else ({}, True)

        return {}, True


class MismatchParametersSolver:
    """Implement the mismatch parameters solver component."""

    def __init__(self) -> None:
        self.shape_simulator = ShapeSimulator()
        self.candidate_adapter = MismatchCandidateAdapter()

    def build_solver_inputs(self, step2_result: dict[str, Any]) -> dict[str, Any]:
        """Build solver inputs."""
        root_cause = step2_result.get("root_cause") or {}
        shape_pairs = []
        for shape_pair in BuggyShapeModeler(step2_result).extract_target_layer_io_shape_pairs():
            shape_pairs.append(
                {
                    **shape_pair,
                    "buggy_api_call_index": shape_pair.get("target_layer_index"),
                }
            )

        return {
            "buggy_api_call_shape_pairs": shape_pairs,
            "source_code": step2_result.get("final_code", ""),
            "merge_context": root_cause.get("merge_context") or {},
        }

    def validate_candidate(
        self,
        candidate: dict[str, Any],
        solver_inputs: dict[str, Any],
        buggy_api_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate candidate."""
        masked_param_names = self.candidate_adapter.get_masked_param_names(
            buggy_api_call.get("masked_api_call", "")
        )
        masked_candidate = {
            name: value
            for name, value in candidate.items()
            if name in masked_param_names
        }

        shape_pair = find_shape_pair(
            solver_inputs["buggy_api_call_shape_pairs"],
            buggy_api_call,
        )
        if shape_pair is None:
            return {"is_valid": False, "reason": "missing_shape_pair"}

        input_shape = shape_pair.get("modeled_input_shape")
        oracle_shape = shape_pair.get("oracle_output_shape")

        if input_shape is None or oracle_shape is None:
            return {"is_valid": False, "reason": "missing_input_or_oracle_shape"}

        api_name = resolve_api_name(
            buggy_api_call=buggy_api_call,
            source_code=solver_inputs["source_code"],
            supported_api_names=set(self.shape_simulator.API_TO_OP),
        )

        params = extract_call_params(buggy_api_call, api_name)
        params.update(masked_candidate)

        try:
            simulated_shape = self.shape_simulator.simulate(
                api_name=api_name,
                input_shape=input_shape,
                **params,
            )

        except Exception as exc:
            return {
                "is_valid": False,
                "reason": "simulation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        ignored_axes = resolve_merge_ignored_axes(
            solver_inputs.get("merge_context") or {},
            rank=len(simulated_shape),
        )
        is_valid = assert_same_shape(
            simulated_shape,
            oracle_shape,
            ignored_axes=ignored_axes,
        )

        return {
            "is_valid": is_valid,
            "reason": "shape_matched" if is_valid else "shape_mismatch",
            "buggy_api_call_index": buggy_api_call.get("_buggy_api_call_index"),
            "masked_api_call": buggy_api_call["masked_api_call"],
            "input_shape": input_shape,
            "target_output_shape": oracle_shape,
            "simulated_output_shape": simulated_shape,
            "solved_params": masked_candidate,
        }

    def validate_candidate_for_all_buggy_api_calls(
        self,
        candidate: dict[str, Any],
        solver_inputs: dict[str, Any],
        buggy_api_calls: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Validate candidate for all buggy api calls."""
        solved_buggy_api_calls = []

        for buggy_api_call in buggy_api_calls:
            api_call_candidate, is_structured_candidate = self.candidate_adapter.extract_candidate_for_buggy_api_call(
                candidate=candidate,
                buggy_api_call=buggy_api_call,
            )
            if is_structured_candidate and not api_call_candidate:
                logger.info("missing_candidate_for_buggy_api_call")
                return None

            validation_result = self.validate_candidate(
                candidate=api_call_candidate,
                solver_inputs=solver_inputs,
                buggy_api_call=buggy_api_call,
            )
            if not validation_result["is_valid"]:
                logger.info(validation_result.get("reason", "buggy_api_call_mismatch"))
                return None

            solved_api_call = validation_result.copy()
            solved_api_call.pop("is_valid", None)
            solved_api_call.pop("reason", None)
            solved_buggy_api_calls.append(solved_api_call)

        return solved_buggy_api_calls

    def main(
        self,
        step2_result: dict[str, Any],
        candidate_space: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the main workflow."""
        solver_inputs = self.build_solver_inputs(step2_result)
        buggy_api_calls = self.candidate_adapter.get_buggy_api_calls(step2_result)
        candidates = self.candidate_adapter.enumerate_candidate_combinations(
            candidate_space or []
        )

        if not buggy_api_calls:
            logger.info("no_buggy_api_call")
        elif not candidates:
            logger.info("no_candidate")
        else:
            for candidate in candidates:
                validation_result = self.validate_candidate_for_all_buggy_api_calls(
                    candidate=candidate,
                    solver_inputs=solver_inputs,
                    buggy_api_calls=buggy_api_calls,
                )

                if validation_result is not None:
                    logger.info("solved_params_success")
                    return validation_result

            logger.info("no_valid_candidate")

        return []


