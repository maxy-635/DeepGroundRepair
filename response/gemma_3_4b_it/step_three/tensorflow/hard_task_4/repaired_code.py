import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding='same')
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')
        self.reshape = tf.keras.layers.Reshape((10,))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.depthwise_conv(x)
        x = self.global_avg_pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)