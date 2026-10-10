import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer to adjust feature dimensionality to 16
        self.conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # First level residual block (basic block)
        self.basic_block = self._build_basic_block()

        # Second level residual blocks (two blocks)
        self.residual_block1 = self._build_residual_block()
        self.residual_block2 = self._build_residual_block()

        # Third level global branch
        self.global_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Final layers
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def _build_basic_block(self):
        """Builds a basic block with main path and branch"""
        model = tf.keras.Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1),
            ReLU()
        ])
        return model

    def _build_residual_block(self):
        """Builds a residual block with main path and branch"""
        # Main path
        main_path = tf.keras.Sequential([
            self._build_basic_block(),
            Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            BatchNormalization(axis=-1)
        ])

        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
                       kernel_initializer='zeros')

        # Combine paths
        model = tf.keras.Sequential([
            Add(),
            ReLU()
        ])

        return model

    def call(self, inputs):
        # Initial convolution
        x = self.conv1(inputs)

        # First level residual connection
        x = self.basic_block(x)
        x = Add()([x, inputs])  # Skip connection

        # Second level residual connections
        x = self.residual_block1(x)
        x = self.residual_block2(x)

        # Third level global branch
        global_branch = self.global_conv(inputs)
        x = Add()([x, global_branch])

        # Final layers
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor (CIFAR-10 images are 32x32 with 3 channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)