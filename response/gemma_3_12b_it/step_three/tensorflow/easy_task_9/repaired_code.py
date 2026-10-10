import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Flatten, Dense, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # 1x1 Convolutional layer to increase dimensionality
        self.conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_increase')

        # 3x3 Depthwise Separable Convolutional layer for feature extraction
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu', name='depthwise_conv')

        # 1x1 Convolutional layer to reduce dimensionality
        self.conv1x1_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_reduce')

        # Add layer to combine the processed output with the original input
        self.add_layer = Add(name='add_layer')

        # Flatten layer
        self.flatten_layer = Flatten(name='flatten_layer')

        # Fully connected layer for classification
        self.dense_layer = Dense(units=10, activation='softmax', name='dense_layer')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # Increase dimensionality
        x = self.conv1x1_increase(inputs)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)

        # Reduce dimensionality
        x = self.conv1x1_reduce(x)

        # Add the processed output to the original input
        x = self.add_layer([inputs, x])

        # Flatten the output
        x = self.flatten_layer(x)

        # Fully connected layer for classification
        output = self.dense_layer(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (28x28 grayscale image)
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Batch size 1, 28x28 image, 1 channel
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)