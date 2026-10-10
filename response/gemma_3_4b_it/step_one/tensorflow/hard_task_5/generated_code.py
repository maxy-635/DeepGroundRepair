import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.block1_conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.block1_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        self.block2_reshape = tf.keras.layers.Reshape(target_shape=(64, 64, 3, 32))
        self.permute1 = tf.keras.layers.Permute([3, 4])
        self.reshape2 = tf.keras.layers.Reshape(target_shape=(64, 64, 32))

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)
        self.branch = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')

        self.add = tf.keras.layers.Add()
        self.dense = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        # Block 1
        x = self.block1_conv1(inputs)
        x = self.block1_conv2(x)
        x = self.block1_conv3(x)
        fused_features = self.concat1([x, x, x])

        # Block 2
        reshaped_features = self.reshape2(fused_features)
        permuted_features = self.permute1(reshaped_features)
        reshaped_features_back = self.reshape2(permuted_features)

        # Block 3
        depthwise_output = self.depthwise_conv(reshaped_features_back)
        branch_output = self.branch(inputs)
        combined_output = self.add([depthwise_output, branch_output])

        # Fully Connected Layer
        output = self.dense(combined_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 64, 64, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)