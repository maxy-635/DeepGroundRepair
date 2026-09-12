import ast


class SelfModuleCallExtractor:

    def parse_crash_stmt(self, crash_code: str):
        """Parse crash stmt."""
        try:
            tree = ast.parse(crash_code)
        except SyntaxError:
            return None

        if not tree.body:
            return None

        return tree.body[0]

    def extract_self_attribute_name(self, node: ast.AST):

        if not isinstance(node, ast.Attribute):
            return None
        if not isinstance(node.value, ast.Name):
            return None
        if node.value.id != "self":
            return None
        return node.attr

    def extract_call_node(self, stmt: ast.stmt):

        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            return stmt.value
        return None

    def extract_self_module_from_call_tree(self, call_node: ast.Call):

        for node in ast.walk(call_node):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name):
                continue
            if func.value.id != "self":
                continue

            return func.attr

        return None

    def main(self, crash_code: str):

        stmt = self.parse_crash_stmt(crash_code)
        if stmt is None:
            return None

        call_node = self.extract_call_node(stmt)
        if call_node is None:
            return None

        return self.extract_self_module_from_call_tree(call_node)

    def is_local_forward_call(self, crash_code: str):
        stmt = self.parse_crash_stmt(crash_code)
        if stmt is None:
            return False

        call_node = self.extract_call_node(stmt)
        if call_node is None:
            return False

        func = call_node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                return False
            return True

        if isinstance(func, ast.Name):
            return True

        return False

    def extract_from_init_assignment(self, crash_code: str):
        """Extract from init assignment."""

        stmt = self.parse_crash_stmt(crash_code)
        if stmt is None:
            return None

        if not isinstance(stmt, ast.Assign):
            return None

        for target in stmt.targets:
            module_name = self.extract_self_attribute_name(target)
            if module_name is not None:
                return module_name

        return None
