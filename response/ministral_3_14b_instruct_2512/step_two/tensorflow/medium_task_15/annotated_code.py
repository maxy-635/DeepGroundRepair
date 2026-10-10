import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = Activation('relu')

        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)

        self.dense1 = Dense(units=64)
        self.dense2 = Dense(units=64)

        self.reshape = Reshape((1, 1, 64))
        self.multiply = Multiply()

        self.concatenate = Concatenate(axis=-1)

        self.conv1x1 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.flatten = Flatten()
        self.fc = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Compress feature maps
        compressed = self.global_avg_pool(x)
        compressed = self.dense1(compressed)
        compressed = self.dense2(compressed)

        # Reshape and multiply
        compressed = self.reshape(compressed)
        weighted_features = self.multiply([x, compressed])

        # Concatenate with input
        concatenated = self.concatenate([inputs, weighted_features])

        # Reduce dimensionality and downsample
        x = self.conv1x1(concatenated)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((32, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)