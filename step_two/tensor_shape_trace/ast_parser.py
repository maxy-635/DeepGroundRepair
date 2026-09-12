import ast
from collections import defaultdict, namedtuple, deque, OrderedDict
from step_two.utils import find_parameter_end


class VisitorHelper:
    """Implement the visitor helper component."""

    @staticmethod
    def attribute_name(node):
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            attrs.append(node.id)
            return ".".join(attrs[::-1])
        return None

    @staticmethod
    def collect_assign_target(visitor, node, targets):
        """Collect assign target."""
        if isinstance(node, ast.Name):
            targets.append(visitor.var(node.id, node.lineno, node.col_offset))
        elif isinstance(node, ast.Attribute):
            name = VisitorHelper.attribute_name(node)
            if name is not None:
                targets.append(visitor.var(name, node.lineno, node.col_offset))
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                VisitorHelper.collect_assign_target(visitor, elt, targets)

    @staticmethod
    def create_edge_by_lineno(visitor, load):
        """Create edge by lineno."""
        for stores in visitor.store[-1].values():
            for store in stores:
                if store.lineno == load.lineno:
                    visitor.graph[store].append(load)

    @staticmethod
    def create_edge_by_name(visitor, load):
        """Create edge by name."""
        for store_dict in visitor.store[::-1]:
            for _name, stores in store_dict.items():
                for store in stores:
                    if store.lineno != load.lineno and store.name == load.name:
                        visitor.graph[load].append(store)

    @staticmethod
    def parse_raw_param(visitor, start_lineno, start_index):
        """Parse raw param."""
        result = find_parameter_end(visitor.source_lines, start_lineno, start_index)
        if result is None:
            line = visitor.source_lines[start_lineno - 1]
            end_lineno, end_index = start_lineno, len(line.rstrip("\n"))
        else:
            end_lineno, end_index = result

        if start_lineno == end_lineno:
            raw_param_name = visitor.source_lines[start_lineno - 1][start_index:end_index]
        else:
            raw_param_name = visitor.source_lines[start_lineno - 1][start_index:]
            i = start_lineno
            while i < end_lineno - 1:
                raw_param_name += visitor.source_lines[i]
                i += 1
            raw_param_name += visitor.source_lines[end_lineno - 1][:end_index]

        return visitor.raw_param(raw_param_name, start_lineno, start_index, end_lineno, end_index)

    @staticmethod
    def callable_name(node):
        if isinstance(node, ast.Call):
            return "[call]"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return VisitorHelper.attribute_name(node)
        if isinstance(node, ast.Subscript):
            try:
                return ast.unparse(node)
            except Exception:
                return "[subscript]"
        return None


class Visitor(ast.NodeVisitor):
    def __init__(self, source_lines):
        """Initialize the instance."""
        self.source_lines = source_lines
        self.source_text = "".join(source_lines)

        self.lineno_varname = OrderedDict()
        self.lineno_function_call = OrderedDict()

        self.var = namedtuple("variable", ["name", "lineno", "start_index"])
        self.raw_param = namedtuple("raw_param", ["name", "start_lineno", "start_index", "end_lineno", "end_index"])
        self.func_key = namedtuple("func_key", ["name", "lineno", "start_index"])

        self.func_key_var_params = defaultdict(list)
        self.func_key_raw_params = defaultdict(list)
        self.func_key_return_value = defaultdict(list)

        self.graph = defaultdict(list)
        self.store = [defaultdict(list)]
        self.load = [{}]

        self._current_called_function_key = []
        self._current_attributes = []
        self._class_stack = []
        self._function_stack = []

    def _should_record_runtime_shape_target(self):
        return bool(self._class_stack and self._function_stack)

    def visit_ClassDef(self, node):
        """Visit class def."""
        self._class_stack.append(node.name)
        try:
            super().generic_visit(node)
        finally:
            self._class_stack.pop()

    def _attribute_name(self, node):
        return VisitorHelper.attribute_name(node)

    def _collect_assign_target(self, node, targets):
        """Collect assign target."""
        VisitorHelper.collect_assign_target(self, node, targets)

    def visit_FunctionDef(self, node):
        """Visit function def."""
        self._function_stack.append(node.name)
        self.store.append(defaultdict(list))
        self.load.append({})
        try:
            for arg in node.args.args:
                self.store[-1][arg.arg].append(self.var(arg.arg, arg.lineno, arg.col_offset))
            super().generic_visit(node)
        finally:
            self.store.pop()
            self.load.pop()
            self._function_stack.pop()

    def visit_Attribute(self, node):
        """Visit attribute."""
        self._current_attributes.append(node.attr)
        super().generic_visit(node)
        self._current_attributes = []

    def visit_Name(self, node):
        """Visit name."""
        if self._current_attributes:
            attribute = ".".join(self._current_attributes[::-1])
            name = node.id + "." + attribute
        else:
            name = node.id

        v = self.var(name, node.lineno, node.col_offset)
        if isinstance(node.ctx, ast.Store):
            self.store[-1][name].append(v)
            if self._should_record_runtime_shape_target():
                if node.lineno not in self.lineno_varname:
                    self.lineno_varname[node.lineno] = []
                self.lineno_varname[node.lineno].append(name)
        elif isinstance(node.ctx, ast.Load):
            self.load[-1][name] = v
            self.create_edge_by_lineno(v)
            self.create_edge_by_name(v)
        elif isinstance(node.ctx, ast.Param):
            self.load[-1][name] = v

        if self._current_called_function_key and self._current_called_function_key[-1] != v:
            self.func_key_var_params[self._current_called_function_key[-1]].append(v)

    def create_edge_by_lineno(self, load):
        """Create edge by lineno."""
        VisitorHelper.create_edge_by_lineno(self, load)

    def create_edge_by_name(self, load):
        """Create edge by name."""
        VisitorHelper.create_edge_by_name(self, load)

    def query_data_dependency(self, variable):
        """Query data dependency."""
        visited = [variable]
        children = deque(self.graph[variable])
        if not children:
            return []

        while children:
            variable = children.popleft()
            if variable not in visited:
                children.extend(deque(self.graph[variable]))
                visited.append(variable)

        return visited

    def visit_Assign(self, node):
        """Visit assign."""
        targets = []
        for t in node.targets:
            self._collect_assign_target(t, targets)

        for t in node.targets:
            super().visit(t)

        if isinstance(node.value, ast.Call):
            function_key = self.visit_Call(node.value)
            self.func_key_return_value[function_key] = targets
        else:
            super().visit(node.value)

    def visit_AugAssign(self, node):
        """Visit aug assign."""
        targets = []
        self._collect_assign_target(node.target, targets)

        super().visit(node.target)

        for target in targets:
            self._connect_augassign_target_to_previous_store(target)

        super().visit(node.value)

    def _connect_augassign_target_to_previous_store(self, target):
        """Connect augassign target to previous store."""
        for store_dict in self.store[::-1]:
            stores = store_dict.get(target.name, [])
            for store in reversed(stores):
                if store == target:
                    continue
                if store.lineno > target.lineno:
                    continue
                self.graph[target].append(store)
                return

    def parse_raw_param(self, start_lineno, start_index):
        """Parse raw param."""
        return VisitorHelper.parse_raw_param(self, start_lineno, start_index)

    def visit_Call(self, node):
        """Visit call."""
        function_name = VisitorHelper.callable_name(node.func)
        if function_name is None:
            function_name = "[unknown]"

        function_key = self.func_key(function_name, node.lineno, node.col_offset)
        if node.lineno not in self.lineno_function_call:
            self.lineno_function_call[node.lineno] = []
        self.lineno_function_call[node.lineno].append(function_key)
        self._current_called_function_key.append(function_key)

        for arg in node.args:
            raw_param_name = ast.get_source_segment(self.source_text, arg)
            if raw_param_name is None:
                start_index = arg.col_offset - 1 if isinstance(arg, ast.Tuple) else arg.col_offset
                raw_param = self.parse_raw_param(arg.lineno, start_index)
            else:
                raw_param = self.raw_param(
                    raw_param_name, arg.lineno, arg.col_offset, arg.end_lineno, arg.end_col_offset
                )
            self.func_key_raw_params[function_key].append((None, raw_param))

        for keyword in node.keywords:
            karg, kvalue = keyword.arg, keyword.value
            raw_param_name = ast.get_source_segment(self.source_text, kvalue)
            if raw_param_name is None:
                start_index = kvalue.col_offset - 1 if isinstance(kvalue, ast.Tuple) else kvalue.col_offset
                raw_param = self.parse_raw_param(kvalue.lineno, start_index)
            else:
                raw_param = self.raw_param(
                    raw_param_name,
                    kvalue.lineno,
                    kvalue.col_offset,
                    kvalue.end_lineno,
                    kvalue.end_col_offset,
                )
            self.func_key_raw_params[function_key].append((karg, raw_param))

        super().generic_visit(node)
        last = self._current_called_function_key.pop()
        if self._current_called_function_key:
            self.func_key_var_params[self._current_called_function_key[-1]].append(last)
        return function_key
