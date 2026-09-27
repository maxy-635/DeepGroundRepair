import os
import runpy
import functools
import inspect
import ast
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



class NoWeightRunner:
    """Implement the no weight runner component."""

    RUN_TARGET_ENV = "NO_WEIGHT_VALIDATION_TARGET"

    class _DummyLoadStatus:
        """Implement the dummy load status component."""

        def expect_partial(self):
            return self

        def assert_consumed(self):
            return self

        def assert_existing_objects_matched(self):
            return self

    class _DummyIncompatibleKeys:
        """Implement the dummy incompatible keys component."""

        missing_keys = []
        unexpected_keys = []

    class _DummyVisionDataset:
        """Implement the dummy vision dataset component."""

        def __init__(
            self,
            image_shape,
            length=4,
            num_classes=10,
            transform=None,
            target_transform=None,
            **kwargs,
        ):
            self.image_shape = image_shape
            self.length = length
            self.num_classes = num_classes
            self.transform = transform
            self.target_transform = target_transform
            self.classes = [str(index) for index in range(num_classes)]
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
            self.targets = [0 for _ in range(length)]
            self.labels = self.targets

            try:
                import numpy as np

                self.data = np.zeros((length, *image_shape), dtype="float32")
            except Exception:
                self.data = None

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            try:
                import numpy as np

                image = np.zeros(self.image_shape, dtype="float32")
            except Exception:
                image = 0

            target = self.targets[index % self.length]

            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return image, target

    @staticmethod
    def _patch_module_callables(module, forced_kwargs):
        """Patch module callables."""

        for name in dir(module):
            if name.startswith("_"):
                continue

            try:
                obj = getattr(module, name)
            except Exception:
                continue

            if not callable(obj):
                continue

            try:
                signature = inspect.signature(obj)
            except (TypeError, ValueError):
                continue

            active_forced_kwargs = {
                key: value
                for key, value in forced_kwargs.items()
                if key in signature.parameters
            }
            if not active_forced_kwargs:
                continue

            @functools.wraps(obj)
            def wrapped(*args, __fn=obj, __forced_kwargs=active_forced_kwargs, **kwargs):
                kwargs.update(__forced_kwargs)
                return __fn(*args, **kwargs)

            try:
                setattr(module, name, wrapped)
            except Exception:
                continue

    @staticmethod
    def _make_keras_dataset_loader(image_shape, label_shape):
        """Create keras dataset loader."""

        def no_load_data(*args, **kwargs):
            import numpy as np

            sample_count = 4
            x_train = np.zeros((sample_count, *image_shape), dtype="float32")
            x_test = np.zeros((sample_count, *image_shape), dtype="float32")

            if label_shape:
                y_shape = (sample_count, *label_shape)
            else:
                y_shape = (sample_count,)

            y_train = np.zeros(y_shape, dtype="int64")
            y_test = np.zeros(y_shape, dtype="int64")
            return (x_train, y_train), (x_test, y_test)

        return no_load_data

    @classmethod
    def _patch_keras_dataset_modules(cls):
        """Patch keras dataset modules."""

        dataset_specs = {
            "cifar10": ((32, 32, 3), (1,)),
            "cifar100": ((32, 32, 3), (1,)),
            "mnist": ((28, 28), ()),
            "fashion_mnist": ((28, 28), ()),
        }

        for prefix in ["tensorflow.keras.datasets", "keras.datasets"]:
            try:
                parent_module = __import__(prefix, fromlist=[""])
            except Exception:
                parent_module = None

            for dataset_name, (image_shape, label_shape) in dataset_specs.items():
                try:
                    dataset_module = __import__(
                        f"{prefix}.{dataset_name}",
                        fromlist=[""],
                    )
                    dataset_module.load_data = cls._make_keras_dataset_loader(
                        image_shape,
                        label_shape,
                    )
                    if parent_module is not None:
                        setattr(parent_module, dataset_name, dataset_module)
                except Exception:
                    continue

    @classmethod
    def _make_dummy_dataset_factory(cls, image_shape, num_classes=10):
        """Create dummy dataset factory."""

        def no_dataset(*args, **kwargs):
            return cls._DummyVisionDataset(
                image_shape=image_shape,
                num_classes=num_classes,
                transform=kwargs.get("transform"),
                target_transform=kwargs.get("target_transform"),
            )

        return no_dataset

    @classmethod
    def _patch_vision_dataset_module(cls, module, dataset_specs):
        """Patch vision dataset module."""

        for dataset_name, (image_shape, num_classes) in dataset_specs.items():
            try:
                setattr(
                    module,
                    dataset_name,
                    cls._make_dummy_dataset_factory(image_shape, num_classes),
                )
            except Exception:
                continue

    @classmethod
    def patch_tensorflow_keras_datasets(cls):
        """Patch tensorflow keras datasets."""

        cls._patch_keras_dataset_modules()

    @classmethod
    def patch_paddlepaddle_datasets(cls):
        """Patch paddlepaddle datasets."""

        dataset_specs = {
            "Cifar10": ((3, 32, 32), 10),
            "CIFAR10": ((3, 32, 32), 10),
            "Cifar100": ((3, 32, 32), 100),
            "CIFAR100": ((3, 32, 32), 100),
            "MNIST": ((1, 28, 28), 10),
            "FashionMNIST": ((1, 28, 28), 10),
        }

        try:
            import paddle.vision.datasets as datasets

            cls._patch_vision_dataset_module(datasets, dataset_specs)
        except Exception:
            pass

    @classmethod
    def patch_pytorch_datasets(cls):
        """Patch pytorch datasets."""

        dataset_specs = {
            "CIFAR10": ((32, 32, 3), 10),
            "CIFAR100": ((32, 32, 3), 100),
            "MNIST": ((28, 28), 10),
            "FashionMNIST": ((28, 28), 10),
            "SVHN": ((32, 32, 3), 10),
        }

        try:
            import torchvision.datasets as datasets

            cls._patch_vision_dataset_module(datasets, dataset_specs)
        except Exception:
            pass

    @classmethod
    def _patch_tensorflow_application_modules(cls, forced_kwargs):
        """Patch tensorflow application modules."""

        module_names = [
            "tensorflow.keras.applications",
            "tensorflow.keras.applications.resnet",
            "tensorflow.keras.applications.resnet_v2",
            "tensorflow.keras.applications.efficientnet",
            "tensorflow.keras.applications.efficientnet_v2",
            "tensorflow.keras.applications.convnext",
            "keras.applications",
            "keras.applications.resnet",
            "keras.applications.resnet_v2",
            "keras.applications.efficientnet",
            "keras.applications.efficientnet_v2",
            "keras.applications.convnext",
        ]

        for module_name in module_names:
            try:
                module = __import__(module_name, fromlist=[""])
                cls._patch_module_callables(module, forced_kwargs)
            except Exception:
                continue

    @classmethod
    def patch_tensorflow_keras(cls):
        """Patch tensorflow keras."""

        forced_kwargs = {"weights": None}

        try:
            import tensorflow as tf
        except Exception:
            return

        try:
            cls._patch_module_callables(tf.keras.applications, forced_kwargs)
        except Exception:
            pass

        cls._patch_tensorflow_application_modules(forced_kwargs)

        try:
            import keras

            cls._patch_module_callables(keras.applications, forced_kwargs)
        except Exception:
            pass

        try:
            def no_load_weights(self, *args, **kwargs):
                return cls._DummyLoadStatus()

            tf.keras.Model.load_weights = no_load_weights
        except Exception:
            pass

    @classmethod
    def patch_paddlepaddle(cls):
        """Patch paddlepaddle."""

        try:
            import paddle.vision.models as models
            cls._patch_module_callables(models, {"pretrained": False, "weights": None})
        except Exception:
            pass

        try:
            import paddle

            def no_paddle_load(*args, **kwargs):
                return {}

            def no_set_state_dict(self, state_dict, *args, **kwargs):
                return None

            paddle.load = no_paddle_load
            paddle.nn.Layer.set_state_dict = no_set_state_dict
        except Exception:
            pass

    @classmethod
    def patch_pytorch(cls):
        """Patch pytorch."""

        def no_download_state_dict(*args, **kwargs):
            return {}

        try:
            import torch
            import torch.hub

            def no_torch_load(*args, **kwargs):
                return {}

            def no_load_state_dict(self, state_dict, strict=True, assign=False):
                return cls._DummyIncompatibleKeys()

            torch.hub.load_state_dict_from_url = no_download_state_dict
            torch.load = no_torch_load
            torch.nn.Module.load_state_dict = no_load_state_dict
        except Exception:
            pass

        try:
            import torchvision.models as models
        except Exception:
            return

        forced_kwargs = {
            "weights": None,
            "weights_backbone": None,
            "pretrained": False,
            "pretrained_backbone": False,
        }
        cls._patch_module_callables(models, forced_kwargs)

        for module_name in [
            "torchvision.models.detection",
            "torchvision.models.segmentation",
            "torchvision.models.video",
        ]:
            try:
                module = __import__(module_name, fromlist=[""])
                cls._patch_module_callables(module, forced_kwargs)
            except Exception:
                continue

        try:
            import torch.utils.model_zoo as model_zoo

            model_zoo.load_url = no_download_state_dict
        except Exception:
            pass

    @classmethod
    def _detect_frameworks(cls, pyfile):
        """Detect frameworks."""

        frameworks = []
        framework_names = ("tensorflow", "pytorch", "paddlepaddle")
        path_parts = {part.lower() for part in Path(pyfile).parts}

        for name in framework_names:
            if name in path_parts:
                frameworks.append(name)

        try:
            with open(pyfile, "r", encoding="utf-8") as file:
                tree = ast.parse(file.read(), filename=os.fspath(pyfile))
        except (OSError, SyntaxError):
            return frameworks

        module_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_roots.add(node.module.split(".", 1)[0])

        framework_modules = {
            "tensorflow": ("tensorflow", "keras"),
            "pytorch": ("torch", "torchvision"),
            "paddlepaddle": ("paddle",),
        }
        for name, roots in framework_modules.items():
            if name not in frameworks and any(root in module_roots for root in roots):
                frameworks.append(name)

        return frameworks

    @classmethod
    def run_target(cls, pyfile):
        """Run target."""

        frameworks = cls._detect_frameworks(pyfile)

        patchers = {
            "tensorflow": (
                cls.patch_tensorflow_keras,
                cls.patch_tensorflow_keras_datasets,
            ),
            "pytorch": (
                cls.patch_pytorch,
                cls.patch_pytorch_datasets,
            ),
            "paddlepaddle": (
                cls.patch_paddlepaddle,
                cls.patch_paddlepaddle_datasets,
            ),
        }
        for framework in frameworks:
            for patcher in patchers.get(framework, ()):
                patcher()

        runpy.run_path(str(pyfile), run_name="__main__")


if __name__ == "__main__":
    target_file = os.environ.get(NoWeightRunner.RUN_TARGET_ENV)

    if target_file:
        NoWeightRunner.run_target(target_file)
