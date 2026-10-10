import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPool2D, ReLU, Add, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.main_relu1 = ReLU(max_value=None, negative_slope=0.0)
        self.main_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.main_relu2 = ReLU(max_value=None, negative_slope=0.0)
        self.main_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.main_conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.main_relu3 = ReLU(max_value=None, negative_slope=0.0)
        self.main_conv4 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        self.main_relu4 = ReLU(max_value=None, negative_slope=0.0)
        self.main_pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path
        self.branch_conv = Conv2D(filters=64, kernel_size='mask', strides='mask', padding='mask', activation=None)

        # Final layers
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main path
        x = self.main_conv1(inputs)  # output tensor shape: (1, 28, 28, 32)
        x = self.main_relu1(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.main_conv2(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.main_relu2(x)  # output tensor shape: (1, 28, 28, 32)
        x = self.main_pool(x)  # output tensor shape: (1, 14, 14, 32)

        x = self.main_conv3(x)  # output tensor shape: (1, 14, 14, 64)
        x = self.main_relu3(x)  # output tensor shape: (1, 14, 14, 64)
        x = self.main_conv4(x)  # output tensor shape: (1, 14, 14, 64)
        x = self.main_relu4(x)  # output tensor shape: (1, 14, 14, 64)
        x = self.main_pool2(x)  # output tensor shape: (1, 7, 7, 64)

        # Branch path
        branch = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 7, 7, 64)

        # Sum outputs
        x = self.add([x, branch])

        # Final layers
        x = self.flatten(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # MNIST image shape: (28, 28, 1)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)