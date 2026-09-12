import os
import ast
import sys
from typing import Any

from step_two.root_cause_masking.api_call_rewriter_utils import (
    collect_import_references,
    extract_target_call,
    resolve_runtime_callable
)
from step_one.whoosh_search import WhooshSearch, JiebaAnalyzer
setattr(sys.modules["__main__"], "JiebaAnalyzer", JiebaAnalyzer)


class ShapeKnowledgeRetriever:
    """Implement the shape knowledge retriever component."""

    def __init__(self, dll: str):
        """Initialize the instance."""
        self.dll = dll
        self._searcher = None

    def get_root_cause_api_complete_name(
        self,
        target_layer: dict[str, Any],
        source_code: str,
    ) -> str | None:
        """Return root cause api complete name."""

        import_references = collect_import_references(source_code)
        resolved_modules = {}

        original_call = target_layer.get("original_call")
        if not original_call:
            return None

        try:
            tree = ast.parse(original_call)
        except SyntaxError:
            return None

        if not tree.body:
            return None

        target_call = extract_target_call(tree.body[0])
        if target_call is None:
            return None

        _, resolved_name = resolve_runtime_callable(
            func_node=target_call.func,
            import_references=import_references,
            resolved_modules=resolved_modules,
        )

        root_cause_api_name = resolved_name or ast.unparse(target_call.func)

        return root_cause_api_name

    def _get_searcher(self) -> WhooshSearch:
        """Return searcher."""
        if self._searcher is not None:
            return self._searcher

        index_dir = f"./database/whoosh/whoosh_{self.dll.lower()}"
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Database {self.dll} not found.")

        self._searcher = WhooshSearch(
            index_dir=index_dir,
            docs_path=None,
        )
        return self._searcher

    def retrieve_shape_knowledge(self, root_cause_api_name: str | None) -> dict[str, Any]:
        """Retrieve shape knowledge."""
        if not root_cause_api_name:
            return {}

        retrieved_api_doc = {}
        results = self._get_searcher().search(query_str=root_cause_api_name, limit=1)
        if not results:
            return retrieved_api_doc

        result = results[0]
        retrieved_api_doc['api_name'] = root_cause_api_name
        retrieved_api_doc['api_parameters'] = result.get("api_parameters")
        retrieved_api_doc['tensor_shape_transform_formula'] = result.get("api_shape_formula")

        return retrieved_api_doc

    def main(self, target_layer: dict[str, Any], source_code: str) -> dict[str, Any]:
        """Run the main workflow."""
        root_cause_api_name = self.get_root_cause_api_complete_name(
            target_layer, source_code
        )


        retrieved_api_doc = self.retrieve_shape_knowledge(root_cause_api_name)

        return retrieved_api_doc

