import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, DepthwiseConv2D, SeparableConv2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()
        self.concat1 = Concatenate(axis=-1)
        self.dense1 = Dense(128, activation='relu')
        self.reshape = Reshape(target_shape='mask')

        # Second Block
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concat2 = Concatenate(axis=-1)
        self.flatten4 = Flatten()
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # First Block
        x = self.avg_pool1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x = self.flatten1(x)  # output tensor shape: (1, 784)
        y = self.avg_pool2(inputs)  # output tensor shape: (1, 14, 14, 1)
        y = self.flatten2(y)  # output tensor shape: (1, 196)
        z = self.avg_pool3(inputs)  # output tensor shape: (1, 7, 7, 1)
        z = self.flatten3(z)  # output tensor shape: (1, 49)
        merged = self.concat1([x, y, z])  # output tensor shape: (1, 1029)
        dense_out = self.dense1(merged)  # output tensor shape: (1, 128)
        reshaped_out = self.reshape(dense_out)

        # Second Block
        x1 = self.depthwise_conv1(reshaped_out)
        x2 = self.depthwise_conv2(reshaped_out)
        x3 = self.depthwise_conv3(reshaped_out)
        x4 = self.depthwise_conv4(reshaped_out)
        merged2 = self.concat2([x1, x2, x3, x4])
        flattened = self.flatten4(merged2)
        output = self.dense2(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input: batch size 1, 28x28 image, 1 channel
    output_tensor = model(input_tensor)
    print(output_tensor.shape)