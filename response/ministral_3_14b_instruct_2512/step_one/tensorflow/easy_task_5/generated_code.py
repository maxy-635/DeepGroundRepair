import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose, Reshape, Activation, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # 3x3 convolution to extract features
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # 1x1 convolution to restore dimensionality
        self.conv1x1_restore = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Flatten layer
        self.flatten = Flatten()

        # Fully connected layer for classification
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.conv1x1_reduce(inputs)
        x = self.conv3x3(x)
        x = self.conv1x1_restore(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST images are 28x28 grayscale
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)