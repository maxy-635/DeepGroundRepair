import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Flatten, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DL_Model, self).__init__()
        self.input_shapes = input_shape
        self.num_classes = num_classes

        # Define the block that will be used in both branches
        self.block = self._create_block()

    def _create_block(self):
        """Creates a block that applies global average pooling followed by two dense layers."""
        def block(input_tensor):
            # Global average pooling
            pooled = GlobalAveragePooling2D(keepdims=True)(input_tensor)

            # First dense layer to reduce to channel dimension
            dense1 = Dense(units=self.input_shape[-1], activation='relu')(pooled)

            # Second dense layer to produce weights
            weights = Dense(units=self.input_shape[-1], activation='sigmoid')(dense1)

            # Reshape to match input shape
            weights = Reshape(self.input_shape)(weights)

            # Element-wise multiplication
            output = Multiply()([input_tensor, weights])
            return output
        return block

    def call(self, inputs):
        # Branch 1
        branch1 = self.block(inputs)

        # Branch 2 (same block)
        branch2 = self.block(inputs)

        # Concatenate outputs from both branches
        concatenated = Concatenate(axis=-1)([branch1, branch2])

        # Flatten and add final dense layer
        flattened = Flatten()(concatenated)
        output = Dense(units=self.num_classes, activation='softmax')(flattened)

        return output

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)