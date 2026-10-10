import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, DepthwiseConv2D, SeparableConv2D, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Average pooling layers with different scales
        self.avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten and concatenate the outputs
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten4 = Flatten()
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layer and reshape
        self.dense = Dense(units=128, activation='relu')
        self.reshape = Reshape(target_shape=(4, 4, 8))

        # Second block: Depthwise separable convolutional layers
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv7 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Flatten and final fully connected layer
        self.flatten_final = Flatten()
        self.dense_final = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x1 = self.avg_pool1(inputs)
        x1 = self.flatten1(x1)

        x2 = self.avg_pool2(inputs)
        x2 = self.flatten2(x2)

        x4 = self.avg_pool4(inputs)
        x4 = self.flatten4(x4)

        x = self.concatenate([x1, x2, x4])
        x = self.dense(x)
        x = self.reshape(x)

        # Second block
        x1 = self.depthwise_conv1(x)
        x3 = self.depthwise_conv3(x)
        x5 = self.depthwise_conv5(x)
        x7 = self.depthwise_conv7(x)

        x = self.concatenate([x1, x3, x5, x7])
        x = self.flatten_final(x)
        x = self.dense_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)