"""
Pylint 插件：处理 'tensorflow.keras' 导入的 False Positive
2026-06-03升级：
"""

from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class ImportRewriterChecker(BaseChecker):
    """
    Handle Pylint False Positive for 'tensorflow.keras' imports,
    Rewrite and delegate to Pylint to avoid False Positive like "No module named 'keras' in tensorflow"
    """
    __implements__ = IAstroidChecker

    name = "import-rewriter"
    msgs = {
        "C9001": (
            "False Positive for module of 'tensorflow' imports",
            "false-positive-tensorflow-import",
            "Pylint False Positive for 'tensorflow.keras' imports, automatically rewritten to 'keras'",
        ),
    }

    def __init__(self, linter=None):
        super().__init__(linter)
        self.tensorflow_aliases = {"tensorflow", "tf"}

    def _attribute_parts(self, node):
        parts = []
        current_node = node

        while getattr(current_node, "attrname", None):
            parts.append(current_node.attrname)
            current_node = current_node.expr

        if getattr(current_node, "name", None):
            parts.append(current_node.name)

        return list(reversed(parts))

    def _is_tensorflow_keras_attribute(self, node):
        parts = self._attribute_parts(node)
        if len(parts) != 2:
            return False

        return parts[0] in self.tensorflow_aliases and parts[1] == "keras"

    def visit_import(self, node):
        """
        Remember aliases from imports like: import tensorflow as tf.
        """
        for module_name, alias in node.names:
            if module_name == "tensorflow":
                self.tensorflow_aliases.add(alias or module_name)

    def visit_importfrom(self, node):
        """
        Handle Pylint False Positive for 'tensorflow.keras' imports";
        Example: message "No module named 'keras' in tensorflow".
        We rewrite and delegate to Pylint.
        """
        if node.modname.startswith("tensorflow.keras"):
            # False Positive for module of 'tensorflow' imports
            self.add_message("false-positive-tensorflow-import", node=node)
            # correct the import statement to 'keras'
            node.modname = node.modname.replace("tensorflow.keras", "keras")

    def visit_attribute(self, node):
        """
        Record TensorFlow lazy-loaded keras attributes that are ignored through generated-members.
        Example: tf.keras.Model, tf.keras.layers.Conv2D.
        """
        if self._is_tensorflow_keras_attribute(node):
            self.add_message("false-positive-tensorflow-import", node=node)


def register(linter):
    generated_members = tuple(getattr(linter.config, "generated_members", ()) or ())
    tensorflow_generated_members = (
        "keras",
        "tf.keras",
        "tensorflow.keras",
        "tf.keras.*",
        "tensorflow.keras.*",
    )
    linter.config.generated_members = generated_members + tuple(
        member for member in tensorflow_generated_members
        if member not in generated_members
    )
    linter.register_checker(ImportRewriterChecker(linter))
