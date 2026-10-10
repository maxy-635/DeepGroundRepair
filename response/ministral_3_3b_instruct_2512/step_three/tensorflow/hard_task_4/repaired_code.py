import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.conv2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)
        self.flatten = Flatten()
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 3)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 32, 3)
        channel_attention_weights = self.global_avg_pool(x)  # output tensor shape: (1, 3)
        channel_attention_weights = self.fc1(channel_attention_weights)  # output tensor shape: (1, 64)
        channel_attention_weights = self.fc2(channel_attention_weights)
        channel_attention_weights = Reshape((x.shape[1], x.shape[2]))(channel_attention_weights)
        x = Multiply()([x, channel_attention_weights])
        x = self.conv2(x)
        x = Add()([inputs, x])
        x = self.flatten(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)