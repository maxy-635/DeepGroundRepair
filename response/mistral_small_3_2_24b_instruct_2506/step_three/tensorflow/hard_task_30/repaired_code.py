import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, DepthwiseConv2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: dual-path structure
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.add = Add()

        # Second block: three groups with different kernel sizes
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.add([x, inputs])

        # Second block
        x1 = self.depthwise_conv1(x)
        x2 = self.depthwise_conv2(x)
        x3 = self.depthwise_conv3(x)
        x = self.concatenate([x1, x2, x3])

        # Fully connected layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()