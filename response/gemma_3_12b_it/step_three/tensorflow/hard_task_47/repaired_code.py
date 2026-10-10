import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D, BatchNormalization, Conv2D, Concatenate, AveragePooling2D, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block: Three groups of Depthwise Separable Convolutions
        self.group1 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=1, strides=1, padding='same', depth_multiplier=1, activation='relu', name='dwconv1_1'),
            BatchNormalization(axis=-1, name='bn1_1'),
            Conv2D(filters=16, kernel_size=1, strides=1, padding='same', activation='relu', name='conv1_1')
        ])

        self.group2 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu', name='dwconv2_3'),
            BatchNormalization(axis=-1, name='bn2_3'),
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='conv2_3')
        ])

        self.group3 = tf.keras.Sequential([
            DepthwiseConv2D(kernel_size=5, strides=1, padding='same', depth_multiplier=1, activation='relu', name='dwconv3_5'),
            BatchNormalization(axis=-1, name='bn3_5'),
            Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='conv3_5')
        ])

        # Concatenate the outputs of the three groups
        self.concat1 = Concatenate(axis=-1, name='concat1')

        # Second Block: Three Branches
        self.branch1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='convb1_1'),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='convb1_3')
        ])

        self.branch2 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='convb2_1'),
            Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='relu', name='convb2_7'),
            Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu', name='convb2_1_2'),
            Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', name='convb2_3_2')
        ])

        self.branch3 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid', name='avgpool')
        ])

        # Concatenate the outputs of the three branches
        self.concat2 = Concatenate(axis=-1, name='concat2')

        # Fully Connected Layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First Block
        x1 = self.group1(inputs)  # output tensor shape: (1, 32, 32, 16)
        x2 = self.group2(inputs)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.group3(inputs)  # output tensor shape: (1, 32, 32, 64)
        merged = self.concat1([x1, x2, x3])  # output tensor shape: (1, 32, 32, 112)

        # Second Block
        b1 = self.branch1(merged)  # output tensor shape: (1, 32, 32, 128)
        b2 = self.branch2(merged)  # output tensor shape: (1, 32, 32, 128)
        b3 = self.branch3(merged)  # output tensor shape of this layer should be (1, 32, 32, 112)
        merged2 = self.concat2([b1, b2, b3])

        # Fully Connected Layers
        x = self.dense1(merged2)
        output = self.dense2(x)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)