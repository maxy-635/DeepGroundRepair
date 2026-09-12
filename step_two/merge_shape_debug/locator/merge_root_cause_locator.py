import ast
from loguru import logger
from step_two.crash_locator.root_cause_locator import InitModuleFinder
from step_two.crash_locator.traceback_parser import TracebackParser
from step_two.merge_shape_debug.context.merge_trace_utils import build_merge_context
from step_two.merge_shape_debug.context.merge_trace_utils import build_visitor
from step_two.merge_shape_debug.locator.merge_shape_oracle_builder import MergeShapeOracleBuilder
from step_two.merge_shape_debug.locator.target_api_locator import TargetApiAndParameterLocator
from step_two.merge_shape_debug.locator.target_branch_identifier import TargetBranchIdentifier



class TargetLayerModelingSequenceBuilder:
    """Implement the target layer modeling sequence builder component."""

    def __init__(self, rule_registry=None):
        """Initialize the instance."""
        self.oracle_builder = MergeShapeOracleBuilder()
        self.rule_registry = rule_registry or {}
        self.layer_type_map = self.rule_registry.get("layer_type_map", {})
        self.api_axis_semantics = self.rule_registry.get("api_axis_semantics", {})

    def match_producer_call(self, producer_calls, target_layer):
        """Match producer call."""
        layer_name = target_layer.get("layer_name")
        init_line_no = target_layer.get("line_no")
        call_line_no = target_layer.get("call_line_no")
        for producer_call in producer_calls or []:
            same_layer_name = layer_name and producer_call.get("layer_name") == layer_name
            same_init_line = producer_call.get("init_line_no") == init_line_no
            same_call_line = producer_call.get("call_line_no") == call_line_no
            if same_layer_name or same_init_line or same_call_line:
                return producer_call
        return None

    def find_hint_index(self, target_layer, merge_annotation_hints, target_layer_index=None):
        """Find hint index."""
        if isinstance(target_layer_index, int):
            for index, hint in enumerate(merge_annotation_hints or []):
                if hint.get("target_layer_index") == target_layer_index:
                    return index

        target_layer_line = target_layer.get("line_no")
        if isinstance(target_layer_line, int):
            for index, hint in enumerate(merge_annotation_hints or []):
                if hint.get("target_layer_line_no") == target_layer_line:
                    return index

        layer_name = target_layer.get("layer_name")
        target_call_line = target_layer.get("call_line_no")
        for index, hint in enumerate(merge_annotation_hints or []):
            same_layer_name = layer_name and hint.get("layer_name") == layer_name
            same_call_line = hint.get("line_no") == target_call_line
            if same_layer_name and same_call_line:
                return index

        same_line_indexes = [
            index
            for index, hint in enumerate(merge_annotation_hints or [])
            if hint.get("line_no") == target_call_line
        ]
        if len(same_line_indexes) == 1:
            return same_line_indexes[0]
        if same_line_indexes and not any(
            merge_annotation_hints[index].get("target_layer_index") is not None
            or merge_annotation_hints[index].get("target_layer_line_no") is not None
            or merge_annotation_hints[index].get("layer_name") is not None
            for index in same_line_indexes
        ):
            return same_line_indexes[0]

        return None

    def resolve_upstream_target_layer_index(self, modeling_sequence, current_input_var_names):
        """Resolve upstream target layer index."""
        current_input_var_names = set(current_input_var_names or [])
        if not current_input_var_names:
            return None

        for item in reversed(modeling_sequence):
            output_var_name = item.get("output_var_name")
            if output_var_name and output_var_name in current_input_var_names:
                return item["target_layer_index"]
        return None

    def infer_declared_input_channels(self, target_layer):
        """Infer declared input channels."""
        call_texts = [
            target_layer.get("normalized_call"),
            target_layer.get("original_call"),
            target_layer.get("masked_api_call"),
        ]
        if not any("conv" in str(text).lower() for text in call_texts if text):
            return None

        for call_text in call_texts:
            if not isinstance(call_text, str) or not call_text.strip():
                continue
            try:
                tree = ast.parse(call_text)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "in_channels":
                        continue
                    try:
                        value = ast.literal_eval(keyword.value)
                    except (ValueError, SyntaxError):
                        continue
                    if isinstance(value, int):
                        return value

        return None

    def replace_input_channel_from_layer_signature(self, input_shape, target_layer, merge_context):
        """Replace input channel from layer signature."""
        if input_shape is None:
            return None

        declared_in_channels = self.infer_declared_input_channels(target_layer)
        if declared_in_channels is None:
            return list(input_shape)

        result = list(input_shape)
        tensor_layout = str((merge_context or {}).get("tensor_layout", "")).lower()
        if len(result) == 4 and tensor_layout == "channels_last":
            result[-1] = declared_in_channels
        elif len(result) >= 2:
            result[1] = declared_in_channels

        return result

    def resolve_layer_semantic(self, target_layer):
        """Resolve layer semantic."""
        semantic_type = target_layer.get("layer_semantic")
        if semantic_type:
            return semantic_type

        api_name = target_layer.get("api_name")
        if api_name in self.layer_type_map:
            return self.layer_type_map[api_name]

        call_texts = [
            target_layer.get("normalized_call"),
            target_layer.get("original_call"),
            target_layer.get("masked_api_call"),
        ]
        for name, mapped_semantic in self.layer_type_map.items():
            if any(name in str(text) for text in call_texts if text):
                return mapped_semantic
        return None

    def select_api_output_template(self, target_layer, input_shape, actual_output_shape):
        """Select api output template."""
        semantic_type = self.resolve_layer_semantic(target_layer)
        if semantic_type in self.api_axis_semantics and input_shape is not None:
            return list(input_shape)
        if actual_output_shape is not None:
            return list(actual_output_shape)
        if input_shape is not None:
            return list(input_shape)
        return None

    def project_shape_by_mismatch_type(
        self,
        reference_shape,
        template_shape,
        mismatch_types,
        merge_context,
        semantic_rule=None,
    ):
        """Project shape by mismatch type."""
        if reference_shape is None:
            return None
        if template_shape is None:
            return list(reference_shape)

        api_oracle_shape = list(template_shape)
        mismatch_types = set(mismatch_types or [])
        semantic_rule = semantic_rule or {}

        if not (mismatch_types & {"spatial", "channel"}):
            return list(reference_shape)

        api_oracle_shape = self.oracle_builder.compose_shape_by_mismatch_types(
            template_shape=api_oracle_shape,
            reference_shape=reference_shape,
            mismatch_types=mismatch_types,
            merge_context=merge_context,
        )
        return api_oracle_shape

    def project_api_oracle_to_api_output(
        self,
        branch_oracle_shape,
        input_shape,
        actual_output_shape,
        target_layer,
        merge_context,
    ):
        """Project api oracle to api output."""
        semantic_type = self.resolve_layer_semantic(target_layer)
        semantic_rule = self.api_axis_semantics.get(semantic_type) or {}
        template_shape = self.select_api_output_template(
            target_layer=target_layer,
            input_shape=input_shape,
            actual_output_shape=actual_output_shape,
        )
        return self.project_shape_by_mismatch_type(
            reference_shape=branch_oracle_shape,
            template_shape=template_shape,
            mismatch_types=target_layer.get("mismatch_types") or [],
            merge_context=merge_context,
            semantic_rule=semantic_rule,
        )

    def build(self, target_layers, merge_annotation_hints, target_branch, merge_context=None):
        producer_calls = target_branch.get("producer_calls") or []
        sortable_records = []
        for target_layer_index, target_layer in enumerate(target_layers or []):
            producer_call = self.match_producer_call(
                producer_calls=producer_calls,
                target_layer=target_layer,
            )
            sortable_records.append(
                {
                    "target_layer_index": target_layer_index,
                    "target_layer": target_layer,
                    "producer_call": producer_call,
                }
            )

        sortable_records.sort(
            key=lambda item: (
                item["target_layer"].get("call_line_no", float("inf")),
                item["target_layer"].get("line_no", float("inf")),
            )
        )

        modeling_sequence = []
        for record in sortable_records:
            target_layer = record["target_layer"]
            producer_call = record["producer_call"] or {}
            actual_input_shape = producer_call.get("input_shape")
            if actual_input_shape is None:
                input_shapes = producer_call.get("input_shapes") or []
                if len(input_shapes) == 1:
                    actual_input_shape = input_shapes[0]
            if actual_input_shape is not None:
                actual_input_shape = list(actual_input_shape)

            actual_output_shape = producer_call.get("output_shape")
            if actual_output_shape is not None:
                actual_output_shape = list(actual_output_shape)

            merge_annotation_hint_index = self.find_hint_index(
                target_layer=target_layer,
                merge_annotation_hints=merge_annotation_hints,
                target_layer_index=record["target_layer_index"],
            )
            branch_level_oracle_shape = None
            if merge_annotation_hint_index is not None:
                hint = merge_annotation_hints[merge_annotation_hint_index]
                if hint.get("oracle_shape") is not None:
                    branch_level_oracle_shape = list(hint["oracle_shape"])

            upstream_target_layer_index = self.resolve_upstream_target_layer_index(
                modeling_sequence=modeling_sequence,
                current_input_var_names=producer_call.get("input_var_names") or [],
            )
            modeled_input_shape = list(actual_input_shape) if actual_input_shape is not None else None
            if upstream_target_layer_index is not None:
                upstream_record = next(
                    (
                        item
                        for item in reversed(modeling_sequence)
                        if item["target_layer_index"] == upstream_target_layer_index
                    ),
                    None,
                )
                if upstream_record and upstream_record.get("oracle_output_shape") is not None:
                    modeled_input_shape = list(upstream_record["oracle_output_shape"])

            modeled_input_shape = self.replace_input_channel_from_layer_signature(
                input_shape=modeled_input_shape,
                target_layer=target_layer,
                merge_context=merge_context or {},
            )
            oracle_output_shape = self.project_api_oracle_to_api_output(
                branch_oracle_shape=branch_level_oracle_shape,
                input_shape=modeled_input_shape,
                actual_output_shape=actual_output_shape,
                target_layer=target_layer,
                merge_context=merge_context or {},
            )

            modeling_sequence.append(
                {
                    "target_layer_index": record["target_layer_index"],
                    "layer_name": target_layer.get("layer_name"),
                    "modeled_input_shape": modeled_input_shape,
                    "branch_level_oracle_shape": branch_level_oracle_shape,
                    "oracle_output_shape": oracle_output_shape,
                }
            )

        return modeling_sequence


class MergeMismatchRootCauseLocator:
    """Implement the merge mismatch root cause locator component."""

    def __init__(self, mismatch_rules):
        """Initialize the instance."""

        self.branch_identifier = TargetBranchIdentifier()
        self.api_locator = TargetApiAndParameterLocator(mismatch_rules=mismatch_rules)
        self.oracle_builder = MergeShapeOracleBuilder()
        self.modeling_sequence_builder = TargetLayerModelingSequenceBuilder(
            rule_registry=self.api_locator.rule_registry
        )

    def infer_framework(self, origin_code, source_path):
        """Infer framework."""
        lowered_source = origin_code.lower()
        lowered_path = str(source_path).lower()

        if "import torch" in lowered_source or "torch." in lowered_source or "pytorch" in lowered_path:
            return "pytorch"
        if "import tensorflow" in lowered_source or "tf." in lowered_source or "tensorflow" in lowered_path:
            return "tensorflow"
        if "import paddle" in lowered_source or "paddle." in lowered_source:
            return "paddle"
        return "unknown"

    def infer_tensor_layout(self, merge_context):
        """Infer tensor layout."""
        framework = merge_context.get("framework")
        if framework in {"pytorch", "paddle"}:
            return "channels_first"
        if framework == "tensorflow":
            return "channels_last"
        return "channels_last"

    def locate(self, origin_code, runtime_error_info, traced_shapes, initial_input_shapes, source_path):

        traceback_text = runtime_error_info.get("traceback")
        crash_location = TracebackParser().main(traceback_text)

        visitor = build_visitor(origin_code, source_path)
        init_definitions = InitModuleFinder(origin_code).collect()
        merge_context = build_merge_context(
            visitor=visitor,
            traced_shapes=traced_shapes,
            initial_input_shapes=initial_input_shapes or {},
            crash_location=crash_location,
            init_definitions=init_definitions,
            runtime_error_info=runtime_error_info,
            source_code=origin_code,
        )

        if merge_context is None:
            logger.error("Failed to build merge context.")
            return None

        merge_context["framework"] = self.infer_framework(origin_code, source_path)
        merge_context["tensor_layout"] = self.infer_tensor_layout(merge_context)

        mismatch_types = self.branch_identifier.detect_mismatch_types(merge_context)
        if not mismatch_types:
            logger.error("Failed to detect mismatch types.")
            return None

        target_branch = self.branch_identifier.select_target_branch(
            branches=merge_context["branches"],
            mismatch_types=mismatch_types,
            merge_context=merge_context,
        )
        if target_branch is None:
            logger.error("Failed to select target branch.")
            return None

        target_layers = self.api_locator.locate_target_layers_on_branch(
            target_branch=target_branch,
            mismatch_types=mismatch_types,
            init_definitions=init_definitions,
            framework=merge_context.get("framework"),
            source_code=origin_code,
        )
        if not target_layers:
            logger.error("Failed to locate target layers on branch.")
            return None

        target_branch_index = next(
            (
                index
                for index, branch in enumerate(merge_context["branches"])
                if branch is target_branch or branch == target_branch
            ),
            None,
        )
        if target_branch_index is None:
            logger.error("Failed to resolve target branch index.")
            return None

        primary_target_layer_index = 0
        merge_annotation_hints = self.oracle_builder.build_merge_annotation_hints(
            target_layers=target_layers,
            target_branch=target_branch,
            merge_context=merge_context,
            traced_shapes=traced_shapes,
        )
        if not merge_annotation_hints:
            logger.error("Failed to build merge annotation hints.")
            return None

        primary_merge_annotation_hint_index = self.modeling_sequence_builder.find_hint_index(
            target_layer=target_layers[primary_target_layer_index],
            merge_annotation_hints=merge_annotation_hints,
            target_layer_index=primary_target_layer_index,
        )
        if primary_merge_annotation_hint_index is None:
            logger.error("Failed to resolve primary merge annotation hint index.")
            return None

        location_result = {
            "crash_location": {
                "crash_stage": getattr(crash_location, "crash_stage", "unknown"),
                "line_no": crash_location.line_no,
                "crash_code": crash_location.crash_code,
            },
            "root_cause_type": "Tensor_Merge_Mismatch",
            "mismatch_types": mismatch_types,
            "merge_context": merge_context,
            "target_branch_index": target_branch_index,
            "target_layers": target_layers,
            "primary_target_layer_index": primary_target_layer_index,
            "merge_annotation_hints": merge_annotation_hints,
            "primary_merge_annotation_hint_index": primary_merge_annotation_hint_index,
            "target_layer_modeling_sequence": self.modeling_sequence_builder.build(
                target_layers=target_layers,
                merge_annotation_hints=merge_annotation_hints,
                target_branch=target_branch,
                merge_context=merge_context,
            ),
        }

        return location_result

