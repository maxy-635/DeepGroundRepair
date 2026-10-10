import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same')
        self.concatenate = tf.keras.layers.concatenate(inputs='mask', axis='mask')
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)
        self.conv_branch = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same')
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        main_path = inputs
        main_path = self.conv1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.conv3(main_path)
        main_path = self.concatenate([main_path, main_path, main_path])

        branch_path = self.depthwise_conv(inputs)
        branch_path = self.conv_branch(branch_path)
        merged = self.add([main_path, branch_path])
        flattened = self.flatten(merged)
        output = self.dense1(flattened)
        output = self.dense2(output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)