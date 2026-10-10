import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First path: 1x1 convolution
        self.conv1x1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Second path: 1x1 + 3x3 + 3x3 convolutions
        self.conv1x1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_path2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_path2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Third path: 1x1 + 3x3 convolution
        self.conv1x1_path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Fourth path: max pooling + 1x1 convolution
        self.maxpool_path4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv1x1_path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.dense = Dense(units=128, activation='relu')
        self.output_layer = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First path
        path1 = self.conv1x1_path1(inputs)

        # Second path
        path2 = self.conv1x1_path2(inputs)
        path2 = self.conv3x3_path2_1(path2)
        path2 = self.conv3x3_path2_2(path2)

        # Third path
        path3 = self.conv1x1_path3(inputs)
        path3 = self.conv3x3_path3(path3)

        # Fourth path
        path4 = self.maxpool_path4(inputs)
        path4 = self.conv1x1_path4(path4)

        # Concatenate all paths
        concatenated = self.concatenate([path1, path2, path3, path4])

        # Flatten and pass through dense layers
        flattened = self.flatten(concatenated)
        dense_output = self.dense(flattened)
        output_tensor = self.output_layer(dense_output)

        return output_tensor

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)