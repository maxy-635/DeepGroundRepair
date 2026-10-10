import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # Global average pooling
        self.gap = GlobalAveragePooling2D()

        # Fully connected layers to match initial feature channel size
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')

        # Reshape to match initial feature size
        self.reshape = Reshape(target_shape=(1, 1, 64))

        # Concatenate with input
        self.concatenate = Concatenate(axis=-1)

        # 1x1 convolution for dimensionality reduction
        self.conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Average pooling
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Final fully connected layer for classification
        self.fc_final = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global average pooling
        x = self.gap(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape to match initial feature size
        x = self.reshape(x)

        # Concatenate with input
        x = self.concatenate([inputs, x])

        # 1x1 convolution
        x = self.conv_1x1(x)

        # Average pooling
        x = self.avg_pool(x)

        # Flatten
        x = Flatten()(x)

        # Final fully connected layer
        output = self.fc_final(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)