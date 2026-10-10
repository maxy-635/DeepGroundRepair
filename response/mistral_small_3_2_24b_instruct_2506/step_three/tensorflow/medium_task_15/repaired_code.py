import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        # Batch normalization and ReLU activation
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = tf.nn.relu
        # Global average pooling
        self.gap = GlobalAveragePooling2D(keepdims=False)
        # Fully connected layers
        self.dense1 = Dense(units=64, activation=None)
        self.dense2 = Dense(units=64, activation=None)
        # Reshape and multiply
        self.reshape = tf.reshape
        self.multiply = tf.math.multiply
        # Concatenate
        self.concatenate = tf.experimental.numpy.concatenate
        # 1x1 convolution and average pooling
        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        # Final fully connected layer for classification
        self.dense3 = Dense(units=10, activation=None)

    def call(self, inputs):
        # Initial feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        initial_features = x

        # Compress features
        x = self.gap(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu1(x)

        # Reshape and multiply
        x = self.reshape(x, shape=(-1, 1, 1, 64))
        x = self.multiply(initial_features, x)

        # Concatenate with input
        x = self.concatenate([initial_features, x], axis=-1)

        # Reduce dimensionality and downsample
        x = self.conv2(x)
        x = self.avg_pool(x)

        # Final classification
        x = tf.keras.layers.Flatten()(x)
        x = self.dense3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor)