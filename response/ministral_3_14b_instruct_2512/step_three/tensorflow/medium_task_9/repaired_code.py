import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.initial_conv = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )

        # Batch normalization and ReLU for initial path
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # First basic block
        self.conv1 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn2 = BatchNormalization(axis=-1)
        self.relu2 = ReLU()

        # Second basic block
        self.conv2 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.bn3 = BatchNormalization(axis=-1)
        self.relu3 = ReLU()

        # Branch convolutional layer
        self.branch_conv = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=None
        )
        self.branch_bn = BatchNormalization(axis=-1)
        self.branch_relu = ReLU()

        # Average pooling and final layers
        self.avg_pool = AveragePooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )
        self.flatten = Flatten()
        self.dense = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # Initial path
        x = self.initial_conv(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # First basic block
        residual = x
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = Add()([x, residual])

        # Branch path
        branch = self.branch_conv(x)
        branch = self.branch_bn(branch)
        branch = self.branch_relu(branch)

        # Combine paths
        x = Add()([x, branch])

        # Final layers
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print (None, 10) for batch size None