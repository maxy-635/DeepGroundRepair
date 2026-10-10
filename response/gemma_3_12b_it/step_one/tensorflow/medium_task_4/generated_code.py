import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Two blocks of convolution + average pooling
        self.conv1_path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_path1')
        self.avgpool1_path1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='avgpool1_path1')
        self.conv2_path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv2_path1')
        self.avgpool2_path1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='avgpool2_path1')

        # Path 2: Single convolutional layer
        self.conv1_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv1_path2')

        # Concatenate and flatten
        self.flatten = Flatten(name='flatten')

        # Fully connected layer
        self.dense = Dense(units=10, activation='softmax', name='dense')

    def call(self, inputs):
        # Path 1
        x_path1 = self.conv1_path1(inputs)
        x_path1 = self.avgpool1_path1(x_path1)
        x_path1 = self.conv2_path1(x_path1)
        x_path1 = self.avgpool2_path1(x_path1)

        # Path 2
        x_path2 = self.conv1_path2(inputs)

        # Concatenate
        merged = concatenate([x_path1, x_path2], axis=-1)

        # Flatten and dense
        x = self.flatten(merged)
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)