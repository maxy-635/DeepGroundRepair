import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, Dense, Add, DepthwiseConv2D, Reshape, Concatenate

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Split into three groups and process with 1x1 convolutions
        self.conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat1 = Concatenate(axis=-1)

        # Block 2: Channel shuffling
        self.reshape2 = Reshape(target_shape=(-1, 32, 32, 3, 32))
        self.permute2 = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))
        self.reshape2_back = Reshape(target_shape=(-1, 32, 32, 96))

        # Block 3: Depthwise separable convolution
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Branch: Direct connection to input
        self.branch = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.add = Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.conv1_1(inputs)
        x2 = self.conv1_2(inputs)
        x3 = self.conv1_3(inputs)
        x = self.concat1([x1, x2, x3])

        # Block 2
        x = self.reshape2(x)
        x = self.permute2(x)
        x = self.reshape2_back(x)

        # Block 3
        x = self.depthwise_conv3(x)

        # Branch
        branch = self.branch(inputs)

        # Combine
        x = self.add([x, branch])
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (None, 10)