import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import sys
from collections import defaultdict, namedtuple
from contextlib import contextmanager

try:
    import torch
except ImportError:
    torch = None

try:
    import paddle
except ImportError:
    paddle = None

try:
    import tensorflow
except ImportError:
    tensorflow = None



class ShapeNormalizer:
    """Implement the shape normalizer component."""

    @staticmethod
    def normalize(shape):
        """Normalize the input value."""
        if shape is None:
            return [1]
        try:
            normalized = list(shape)
            return normalized if normalized else [1]
        except TypeError:
            return [shape]

    @staticmethod
    def shape_of_tensorflow(value):
        """Return the shape of tensorflow values."""
        if tensorflow is None:
            return None
        shape = getattr(value, "shape", None)
        if isinstance(shape, tensorflow.TensorShape):
            return shape.as_list() if shape.dims else [1]
        return None

    @staticmethod
    def shape_of_torch(value):
        """Return the shape of torch values."""
        if torch is None:
            return None
        shape = getattr(value, "shape", None)
        if isinstance(shape, torch.Size):
            return list(shape) if len(shape) > 0 else [1]
        return None

    @staticmethod
    def shape_of_paddle(value):
        """Return the shape of paddle values."""
        if paddle is None:
            return None
        tensor_type = getattr(paddle, "Tensor", None)
        if tensor_type is None or not isinstance(value, tensor_type):
            return None
        shape = getattr(value, "shape", None)
        return ShapeNormalizer.normalize(shape)

    @classmethod
    def shape_of(cls, value):
        """Return the shape of a supported value."""
        tf_shape = cls.shape_of_tensorflow(value)
        if tf_shape is not None:
            return tf_shape

        torch_shape = cls.shape_of_torch(value)
        if torch_shape is not None:
            return torch_shape

        paddle_shape = cls.shape_of_paddle(value)
        if paddle_shape is not None:
            return paddle_shape

        shape = getattr(value, "shape", None)
        if shape is not None:
            return cls.normalize(shape)

        if isinstance(value, (int, float, bool, complex)):
            return [1]

        return ["unknown"]


class ShapeTraceManager:
    """Implement the shape trace manager component."""
    ENTRY_METHOD_NAMES = {"forward", "call"}

    def __init__(self):
        """Initialize the instance."""
        self._missing = object()
        self.VarKey = namedtuple("VarKey", ["path", "lineno", "var_name"])
        self.top_dir = os.path.realpath(os.getcwd())
        self.top_dir_dot = os.path.join(self.top_dir, ".")
        self._reset_trace_state()
        self.filenames = []
        self.filter_filename = self._default_filter_filename
        self.lineno_varname = {}
        self.trace_shape = False

    def _default_filter_filename(self, filename):
        if filename is None:
            return None
        normalized = os.path.realpath(filename)
        if normalized.startswith(self.top_dir_dot):
            return None
        return normalized

    def _reset_trace_state(self):
        """Reset trace state."""
        self.var_dict = {}
        self.initial_input_shapes = {}
        self.var_stack = [{}]
        self.last_lineno_by_frame = {}
        self.f_globals = {}
        self.f_locals = {}
        self._exception_lineno_by_frame = {}

    def _update_frame_namespaces(self, frame):
        """Update frame namespaces."""
        if frame.f_globals:
            self.f_globals = frame.f_globals
        if frame.f_locals:
            self.f_locals = frame.f_locals

    def _push_scope(self):
        """Push scope."""
        self.var_stack.append({})

    def _pop_scope(self):
        """Pop scope."""
        if len(self.var_stack) > 1:
            self.var_stack.pop()
        else:
            self.var_stack[0] = {}

    def _sync_current_scope(self, frame):
        """Synchronize current scope."""
        self.var_stack[-1] = frame.f_locals

    def _resolve_var_value(self, var_name):
        """Resolve var value."""
        for scope_index in range(len(self.var_stack) - 1, -1, -1):
            scope = self.var_stack[scope_index]
            if var_name in scope:
                return scope[var_name]
        if var_name in self.f_locals:
            return self.f_locals[var_name]
        if var_name in self.f_globals:
            return self.f_globals[var_name]
        return self._missing

    def _record_shapes_for_lineno(self, filename, lineno):
        """Record shapes for lineno."""
        if lineno not in self.lineno_varname:
            return

        for var_name in self.lineno_varname[lineno]:
            value = self._resolve_var_value(var_name)
            if value is not self._missing:
                key = self.VarKey(filename, lineno, var_name)
                self.var_dict[key] = ShapeNormalizer.shape_of(value)

    def _record_initial_inputs(self, frame):
        """Record initial inputs."""
        if frame.f_code.co_name not in self.ENTRY_METHOD_NAMES:
            return

        function_lineno = frame.f_code.co_firstlineno
        if function_lineno in self.initial_input_shapes:
            return

        shape_items = []
        for var_name, value in frame.f_locals.items():
            if var_name in {"self", "cls"}:
                continue

            shape = ShapeNormalizer.shape_of(value)
            if shape == ["unknown"]:
                continue

            shape_items.append(
                {
                    "var_name": var_name,
                    "shape": shape,
                }
            )

        if shape_items:
            self.initial_input_shapes[function_lineno] = shape_items

    def configure(self, filenames, lineno_varname, filter_filename=None, trace_shape=False):
        self.filenames = filenames
        self.filter_filename = filter_filename or self._default_filter_filename
        self.lineno_varname = lineno_varname
        self.trace_shape = trace_shape

    def trace_dispatch(self, frame, event, arg, enabled, callback):
        """Trace dispatch."""

        if not enabled:
            return None

        filename = self.filter_filename(frame.f_code.co_filename)
        if filename not in self.filenames:
            return None

        frame.f_trace = callback

        if event == "call":
            self._push_scope()
            self._update_frame_namespaces(frame)
            if filename and self.trace_shape:
                self._sync_current_scope(frame)
                self._record_initial_inputs(frame)
            return callback

        if event == "line":
            self._update_frame_namespaces(frame)
            if filename and self.trace_shape:
                self._sync_current_scope(frame)
                frame_id = id(frame)
                last_lineno = self.last_lineno_by_frame.get(frame_id, -1)
                exception_lineno = self._exception_lineno_by_frame.get(frame_id)
                if last_lineno != -1 and last_lineno != exception_lineno:
                    self._record_shapes_for_lineno(filename, last_lineno)
                if exception_lineno is not None:
                    self._exception_lineno_by_frame.pop(frame_id, None)
                self.last_lineno_by_frame[frame_id] = frame.f_lineno
            return callback

        if event == "return":
            if filename and self.trace_shape:
                frame_id = id(frame)
                last_lineno = self.last_lineno_by_frame.pop(frame_id, -1)
                exception_lineno = self._exception_lineno_by_frame.pop(frame_id, None)
                if last_lineno != -1 and last_lineno != exception_lineno:
                    self._record_shapes_for_lineno(filename, last_lineno)
            self._pop_scope()
            return callback

        if event == "exception":
            if filename and self.trace_shape:
                self._update_frame_namespaces(frame)
                self._sync_current_scope(frame)
                self._exception_lineno_by_frame[id(frame)] = frame.f_lineno
            return callback

        return callback

    def reset(self):
        self._reset_trace_state()
        self.filenames = []
        self.filter_filename = self._default_filter_filename
        self.lineno_varname = {}
        self.trace_shape = False

    def dumps_stats(self):
        """Return stats."""
        result = defaultdict(list)
        for var_key, shape in self.var_dict.items():
            result[var_key.lineno].append(
                {
                    "path": var_key.path,
                    "lineno": var_key.lineno,
                    "var_name": var_key.var_name,
                    "shape": shape,
                }
            )
        return result

    def dumps_initial_inputs(self):
        """Return initial inputs."""
        return dict(self.initial_input_shapes)


class RuntimeShapesCollector:
    """Implement the runtime shapes collector component."""

    def __init__(self):
        """Initialize the instance."""
        self.running = False
        self.manager = ShapeTraceManager()

    @contextmanager
    def collect(self):
        self.running = True
        try:
            yield
        finally:
            self.running = False

    def default_filter_filename(self, filename):
        return self.manager._default_filter_filename(filename)

    def _trace_dispatch(self, frame, event, arg):
        """Trace dispatch."""
        return self.manager.trace_dispatch(frame, event, arg, self.running, self._trace_dispatch)

    def init_shapes_collection(
        self,
        filenames,
        lineno_varname,
        filter_filename=None,
        trace_shape=False,
    ):
        """Initialize shapes collection."""
        self.manager.configure(
            filenames=filenames,
            lineno_varname=lineno_varname,
            filter_filename=filter_filename,
            trace_shape=trace_shape,
        )
        sys.settrace(self._trace_dispatch)

    def stop_shapes_collection(self):
        """Stop shapes collection."""
        sys.settrace(None)

    def reset_shapes_collection(self):
        """Reset shapes collection."""
        self.manager.reset()
        self.stop_shapes_collection()
        self.running = False

    def dumps_stats(self):
        """Return stats."""
        return self.manager.dumps_stats()

    def dumps_initial_inputs(self):
        """Return initial inputs."""
        return self.manager.dumps_initial_inputs()
