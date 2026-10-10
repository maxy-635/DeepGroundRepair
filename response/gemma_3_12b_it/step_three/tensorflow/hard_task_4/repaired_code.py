import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
import tensorflow.keras.backend as K

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial layers
        self.conv1x1 = Conv2D(filters=3 * K.int_shape(tf.random.normal(shape=(1, 32, 32, 3)))[-1],
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='same',
                               activation='relu',
                               name='conv1x1')
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              depth_multiplier=1,
                                              activation='relu',
                                              name='depthwise_conv')

        # Channel Attention Module
        self.gap = GlobalAveragePooling2D(keepdims=False, name='gap')
        self.fc1 = Dense(units=32, activation='relu', name='fc1')  # Assuming 32 channels after conv1x1
        self.fc2 = Dense(units=32, activation=None, name='fc2')
        self.reshape = Reshape((1, 1, 32), name='reshape')  # Assuming 32 channels after conv1x1
        self.multiply = Multiply(name='multiply')

        # Output layers
        self.conv1x1_out = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_out')
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(units=10, activation='softmax', name='dense')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Initial feature extraction
        x = self.conv1x1(inputs)
        x = self.depthwise_conv(x)

        # Channel Attention
        attention_weights = self.gap(x)
        attention_weights = self.fc1(attention_weights)
        attention_weights = self.fc2(attention_weights)
        attention_weights = self.reshape(attention_weights)
        x = self.multiply([x, attention_weights])

        # Output
        x = self.conv1x1_out(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Dummy input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)