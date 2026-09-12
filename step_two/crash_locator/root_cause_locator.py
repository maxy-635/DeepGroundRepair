import ast
from typing import Dict, List
from step_two.crash_locator.traceback_parser import TracebackParser
from step_two.crash_locator.called_module_extractor import SelfModuleCallExtractor
from step_two.crash_locator.container_refiner import SequentialContainerInitDefinitionRefiner
from step_two.crash_locator.root_cause_models import InitModuleDefinition, ModuleUseDefMapping


class InitModuleFinder(ast.NodeVisitor):
    """Implement the init module finder component."""

    def __init__(self, source_code: str):
        """Initialize the instance."""

        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.module_definitions: Dict[str, InitModuleDefinition] = {}

    def collect(self):

        tree = ast.parse(self.source_code)
        self.visit(tree)

        return self.module_definitions

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class def."""

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                self._collect_from_init(item)
                break

    def _collect_from_init(self, init_func: ast.FunctionDef):
        """Collect from init."""

        for stmt in init_func.body:
            if not isinstance(stmt, ast.Assign):
                continue

            module_type = self._extract_module_type(stmt.value)
            if module_type is None:
                continue

            for target in stmt.targets:
                module_name = self._extract_self_attribute_name(target)
                if module_name is None:
                    continue

                code_line = ast.get_source_segment(self.source_code, stmt)
                if code_line is None:
                    code_line = self.source_lines[stmt.lineno - 1]

                self.module_definitions[module_name] = InitModuleDefinition(
                    module_name=module_name,
                    line_no=stmt.lineno,
                    code_line=code_line.strip(),
                    module_type=module_type,
                    end_line_no=getattr(stmt, "end_lineno", stmt.lineno),
                    children=self._extract_container_children(stmt.value, module_name, stmt.lineno),
                )

    def _extract_self_attribute_name(self, node: ast.AST):
        """Extract self attribute name."""

        if not isinstance(node, ast.Attribute):
            return None
        if not isinstance(node.value, ast.Name):
            return None
        if node.value.id != "self":
            return None
        return node.attr

    def _extract_module_type(self, value_node: ast.AST):
        """Extract module type."""

        if not isinstance(value_node, ast.Call):
            return None

        func = value_node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
        return None

    def _extract_container_children(self, value_node: ast.AST, module_name: str, container_line_no: int) -> List[dict]:
        """Extract container children."""
        module_type = self._extract_module_type(value_node)
        if module_type != "Sequential" or not isinstance(value_node, ast.Call):
            return []

        children = []
        child_nodes = self._collect_sequential_child_nodes(value_node)
        for child_index, child_node in enumerate(child_nodes):
            child_type = self._extract_module_type(child_node)
            if child_type is None:
                continue

            child_source = ast.get_source_segment(self.source_code, child_node)
            if child_source is None:
                child_source = self.source_lines[child_node.lineno - 1].strip()

            children.append(
                {
                    "layer_name": f"{module_name}[{child_index}]",
                    "api_name": child_type,
                    "init_line_no": child_node.lineno,
                    "end_line_no": getattr(child_node, "end_lineno", child_node.lineno),
                    "original_call": child_source.strip(),
                    "container_name": module_name,
                    "container_init_line_no": container_line_no,
                    "child_index": child_index,
                }
            )

        return children

    def _collect_sequential_child_nodes(self, value_node: ast.Call) -> List[ast.AST]:
        """Collect sequential child nodes."""
        child_nodes = []
        for arg in value_node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                child_nodes.extend(arg.elts)
                continue
            child_nodes.append(arg)
        return child_nodes


class TracebackToInitMapper:
    """Implement the traceback to init mapper component."""

    def __init__(self):
        """Initialize the instance."""

        self.traceback_parser = TracebackParser()
        self.call_extractor = SelfModuleCallExtractor()
        self.container_refiner = SequentialContainerInitDefinitionRefiner()

    def collect_init_module_definitions(self,source_code: str):
        """Collect init module definitions."""

        return InitModuleFinder(source_code).collect()

    def extract_called_module_name(self, crash_location):
        """Extract called module name."""

        if crash_location.crash_stage == "init":
            return self.call_extractor.extract_from_init_assignment(
                crash_location.crash_code
            )
        if crash_location.crash_stage == "forward":
            return self.call_extractor.main(crash_location.crash_code)

    def main(self, source_code: str, traceback_text: str, runtime_error_info=None):
        """Run the main workflow."""

        crash_location = self.traceback_parser.main(traceback_text)
        called_module_name = self.extract_called_module_name(crash_location)
        if called_module_name is None:
            raise ValueError(
                "Cannot extract target module from crash code: "
                f"{crash_location.crash_code}"
            )

        module_definitions = self.collect_init_module_definitions(source_code)
        init_definition = module_definitions.get(called_module_name)
        if init_definition is None:
            raise ValueError(
                f"Cannot find self.{called_module_name} definition in __init__"
            )
        init_definition = self.container_refiner.refine_container_init_definition(
            init_definition,
            traceback_text,
            runtime_error_info=runtime_error_info,
        )

        return ModuleUseDefMapping(
            crash_location=crash_location,
            called_module_name=called_module_name,
            init_definition=init_definition,
        )
    
