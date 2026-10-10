import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='conv1_1')
        self.depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu', name='depthwise_conv1')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', name='conv1_2')

        self.depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu', name='depthwise_conv2')
        self.conv2_2 = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu', name='conv2_2')

        # Block 2
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax', name='dense')

    def call(self, inputs):
        # Block 1
        x = self.conv1_1(inputs)
        x = self.depthwise_conv1(x)
        x = self.conv1_2(x)

        y = self.depthwise_conv2(inputs)
        y = self.conv2_2(y)

        features = tf.concat([x, y], axis=-1)

        # Block 2
        shape = tf.shape(features)
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        
        num_groups = 4
        channels_per_group = channels // num_groups
        
        reshaped_tensor = tf.reshape(features, [shape[0], height, width, num_groups, channels_per_group])
        
        permuted_tensor = tf.transpose(reshaped_tensor, [0, 1, 3, 2, 4])
        
        channel_shuffled_tensor = tf.reshape(permuted_tensor, [shape[0], height, width, channels])
        
        # Flatten for dense layer
        flattened_tensor = tf.reshape(channel_shuffled_tensor, [-1, height * width * channels])

        # Fully connected layer
        output = self.dense(flattened_tensor)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Dummy input tensor for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)