import json
from step_two.merge_shape_debug.locator.target_api_parameter_masker import TargetApiParameterMasker
from step_two.merge_shape_debug.locator.spatial_dim_processing_helper import SpatialPreservingConvHeuristic
from step_two.merge_shape_debug.locator.target_api_locator_utils import (
    expand_container_calls,
    merge_call_line_no,
    strip_internal_fields,
    unique_list,
)


class TargetApiAndParameterLocator:
    """Implement the target api and parameter locator component."""

    def __init__(self, mismatch_rules):
        """Initialize the instance."""
        with open(mismatch_rules, "r", encoding="utf-8") as file:
            rules = json.load(file)

        self.rule_registry = {
            "layer_type_map": rules.get("LAYER_TYPE_MAP", {}),
            "api_axis_semantics": rules.get("API_AXIS_SEMANTICS", {}),
            "api_parameter_map": rules.get("API_PARAMETER_MAP", {}),
        }

        self.spatial_preserving_conv_heuristic = SpatialPreservingConvHeuristic(
            rule_registry=self.rule_registry
        )
        self.parameter_masker = TargetApiParameterMasker(
            api_parameter_map=self.rule_registry["api_parameter_map"]
        )

    def resolve_params_to_mask(self, api_name, mismatch_type, framework=None):
        """Resolve params to mask."""
        if not api_name:
            return []
        semantic_type = self.rule_registry["layer_type_map"].get(api_name)
        if semantic_type is None:
            return []

        target_param_rule = self.parameter_masker.get_target_params(
            framework=framework,
            mismatch_type=mismatch_type,
            semantic_type=semantic_type,
        )
        return unique_list(target_param_rule["param_names"])

    def try_build_target_for_call(self, call_info, mismatch_type, framework=None, source_code=None):
        api_name = call_info["api_name"]
        semantic_type = self.rule_registry["layer_type_map"].get(api_name)
        params_to_mask = self.resolve_params_to_mask(
            api_name=api_name,
            mismatch_type=mismatch_type,
            framework=framework,
        )
        if not params_to_mask:
            return None

        normalized_call = self.parameter_masker.normalize_positional_arguments(
            original_call=call_info["original_call"],
            source_code=source_code,
        )
        masked_api_call, _, _ = self.parameter_masker.mask_keyword_params(
            original_call=call_info["original_call"],
            api_name=call_info["api_name"],
            concrete_params=params_to_mask,
            source_code=source_code,
        )

        return {
            "params_to_mask": params_to_mask,
            "layer_name": call_info["layer_name"],
            "line_no": call_info["init_line_no"],
            "end_line_no": call_info.get("end_line_no"),
            "api_name": api_name,
            "layer_semantic": semantic_type,
            "original_call": call_info["original_call"],
            "normalized_call": normalized_call,
            "masked_api_call": masked_api_call,
            "call_line_no": call_info.get("call_line_no"),
            "mismatch_types": [mismatch_type],
            "_api_name": call_info["api_name"],
        }

    def find_first_target_for_mismatch(self, effective_calls, mismatch_type, framework=None, source_code=None):
        """Find first target for mismatch."""
        for index, call_info in enumerate(effective_calls):
            target_record = self.try_build_target_for_call(
                call_info=call_info,
                mismatch_type=mismatch_type,
                framework=framework,
                source_code=source_code,
            )
            if target_record is None:
                continue

            if self.spatial_preserving_conv_heuristic.should_skip_preserving_conv(
                effective_calls=effective_calls,
                current_index=index,
                call_info=call_info,
                mismatch_type=mismatch_type,
            ):
                continue

            return target_record
        return None

    def merge_same_layer_targets(self, targets, source_code=None):
        """Merge same layer targets."""
        merged = {}
        for target_record in targets:
            target_init_line_no = target_record["line_no"]
            if target_init_line_no not in merged:
                merged[target_init_line_no] = {
                    "layer_name": target_record["layer_name"],
                    "call_line_no": target_record.get("call_line_no"),
                    "line_no": target_init_line_no,
                    "end_line_no": target_record.get("end_line_no"),
                    "api_name": target_record.get("api_name"),
                    "layer_semantic": target_record.get("layer_semantic"),
                    "original_call": target_record["original_call"],
                    "normalized_call": target_record.get("normalized_call", target_record["original_call"]),
                    "masked_api_call": target_record["masked_api_call"],
                    "mismatch_types": list(target_record["mismatch_types"]),
                    "params_to_mask": list(target_record["params_to_mask"]),
                    "_api_name": target_record["_api_name"],
                }
                continue

            merged[target_init_line_no]["call_line_no"] = merge_call_line_no(
                merged[target_init_line_no].get("call_line_no"),
                target_record.get("call_line_no"),
            )
            merged[target_init_line_no]["mismatch_types"] = unique_list(
                merged[target_init_line_no]["mismatch_types"] + target_record["mismatch_types"]
            )
            merged[target_init_line_no]["params_to_mask"] = unique_list(
                merged[target_init_line_no]["params_to_mask"] + target_record["params_to_mask"]
            )
            masked_api_call, _, _ = self.parameter_masker.mask_keyword_params(
                original_call=merged[target_init_line_no]["original_call"],
                api_name=merged[target_init_line_no]["_api_name"],
                concrete_params=merged[target_init_line_no]["params_to_mask"],
                source_code=source_code,
            )
            merged[target_init_line_no]["masked_api_call"] = masked_api_call

        merged_targets = list(merged.values())
        merged_targets.sort(key=lambda item: item["line_no"])
        return [
            strip_internal_fields(target_record)
            for target_record in merged_targets
        ]

    def locate_target_layers_on_branch(
        self,
        target_branch,
        mismatch_types,
        init_definitions=None,
        framework=None,
        source_code=None,
    ):
        """Locate target layers on branch."""
        effective_calls = expand_container_calls(
            producer_calls=target_branch["producer_calls"],
            init_definitions=init_definitions or {},
        )

        matched_target_records = []
        for mismatch_type in mismatch_types:
            matched_target_record = self.find_first_target_for_mismatch(
                effective_calls=effective_calls,
                mismatch_type=mismatch_type,
                framework=framework,
                source_code=source_code,
            )
            if matched_target_record is not None:
                matched_target_records.append(matched_target_record)

        return self.merge_same_layer_targets(
            targets=matched_target_records,
            source_code=source_code,
        )
