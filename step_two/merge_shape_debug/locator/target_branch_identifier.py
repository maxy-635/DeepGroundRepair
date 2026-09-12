from step_two.merge_shape_debug.analysis.merge_mismatch_detector import MergeMismatchDetector


class TargetBranchIdentifier(MergeMismatchDetector):
    """Implement the target branch identifier component."""

    def __init__(self):
        """Initialize the instance."""
        super().__init__()

    def select_residual_like_target_branch(self, branches, merge_context=None):
        """Select residual like target branch."""
        if len(branches) != 2:
            return None
        if not all(branch.get("role") == "inputs" for branch in branches):
            return None

        merge_kind = str((merge_context or {}).get("merge_kind", "")).lower()
        if merge_kind != "add":
            return None

        connected_inputs = [set(branch.get("connected_forward_inputs") or []) for branch in branches]
        if not connected_inputs[0] or connected_inputs[0] != connected_inputs[1]:
            return None

        transformed_branches = [branch for branch in branches if branch.get("producer_calls")]
        passthrough_branches = [branch for branch in branches if not branch.get("producer_calls")]
        if len(transformed_branches) != 1 or len(passthrough_branches) != 1:
            return None

        transformed_branch = transformed_branches[0]
        transformed_branch["role"] = "main-path"
        passthrough_branches[0]["role"] = "inputs"

        return transformed_branch

    def build_majority_profile(self, branches, mismatch_types, merge_context=None):
        """Build majority profile."""
        return self.shape_compare.build_majority_profile(branches, mismatch_types, merge_context)

    def compute_deviation_score(self, shape, majority_profile, merge_context=None):
        """Compute deviation score."""
        return self.shape_compare.compute_deviation_score(shape, majority_profile, merge_context)

    def select_accumulator_target_branch(self, branches, merge_context=None):
        """Select accumulator target branch."""
        if str((merge_context or {}).get("statement_kind", "")) != "augassign":
            return None

        accumulator_branches = [
            branch for branch in branches if branch.get("is_accumulator_branch")
        ]
        if len(accumulator_branches) != 1:
            return None

        accumulator_branch = accumulator_branches[0]
        if accumulator_branch.get("producer_calls"):
            return accumulator_branch
        
        return None

    def select_target_branch(self, branches, mismatch_types, merge_context=None):
        """Select target branch."""
        if len(branches) > 2:
            candidates = []
            for branch in branches:
                if branch["role"] == "inputs":
                    continue
                if not branch["shape"]:
                    continue
                candidates.append(branch)

            if not candidates:
                return None

            majority_profile = self.build_majority_profile(candidates, mismatch_types, merge_context)
            if not majority_profile:
                return min(candidates, key=lambda item: item["path_length"])

            best_branch = None
            best_score = -1
            for branch in candidates:
                score = self.compute_deviation_score(branch["shape"], majority_profile, merge_context)
                branch["deviation_score"] = score
                if score > best_score:
                    best_branch = branch
                    best_score = score
                elif score == best_score and best_branch is not None:
                    if branch["path_length"] < best_branch["path_length"]:
                        best_branch = branch

            return best_branch

        if len(branches) == 2:
            non_input_branches = [branch for branch in branches if branch["role"] != "inputs"]

            if len(non_input_branches) == 1:
                return non_input_branches[0]
            
            accumulator_branch = self.select_accumulator_target_branch(branches, merge_context)
            if accumulator_branch is not None:
                return accumulator_branch

            if len(non_input_branches) == 2:
                return min(non_input_branches, key=lambda item: item["path_length"])

            rescued_branch = self.select_residual_like_target_branch(branches, merge_context)
            if rescued_branch is not None:
                return rescued_branch

            return None
