import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First feature extraction path: 1x1 convolution
        self.conv1x1_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Second feature extraction path: sequence of convolutions (1x1, 1x7, 7x1)
        self.conv1x1_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x7_path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')
        self.conv7x1_path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs of the two paths
        self.concatenate = Concatenate(axis=-1)

        # 1x1 convolution to align dimensions
        self.conv1x1_align = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch connection
        self.add = Add()

        # Fully connected layers for classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First path
        path1_output = self.conv1x1_path1(inputs)

        # Second path
        path2_output = self.conv1x1_path2(inputs)
        path2_output = self.conv1x7_path2(path2_output)
        path2_output = self.conv7x1_path2(path2_output)

        # Concatenate and align dimensions
        concatenated = self.concatenate([path1_output, path2_output])
        aligned_output = self.conv1x1_align(concatenated)

        # Add the branch connection
        output = self.add([aligned_output, inputs])

        # Classification
        output = self.flatten(output)
        output = self.dense1(output)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)