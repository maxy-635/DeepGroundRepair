import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, LayerNormalization, Add, Flatten, Dense, Activation

class DL_Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Main path
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')
        self.layer_norm = LayerNormalization(axis=-1)
        self.pointwise_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')
        self.pointwise_conv2 = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Branch path (identity connection)
        self.add = Add()

        # Classification head
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Main path
        x = self.depthwise_conv(inputs)  # output tensor shape: (32, 32, 32, 3)
        x = self.layer_norm(x)  # output tensor shape: (32, 32, 32, 3)
        x = self.pointwise_conv1(x)  # output tensor shape: (32, 32, 32, 64)
        x = self.pointwise_conv2(x)  # output tensor shape of this layer should be (32, 32, 32, 3)

        # Branch path (identity connection)
        branch = inputs  # output tensor shape: (32, 32, 32, 3)

        # Combine paths
        combined = self.add([x, branch])

        # Classification head
        x = self.flatten(combined)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch size 32, 32x32x32 feature maps with 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)