import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.block1_branch = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')
        ])
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.reshape1 = tf.keras.layers.Reshape((64, 64, 1, 32))
        self.permute1 = tf.keras.layers.Permute([3, 4])
        self.reshape2 = tf.keras.layers.Reshape((64, 64, 32))
        self.channel_shuffle = tf.keras.layers.Reshape((64, 64, 32))
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        path_features = self.block1_path(inputs)
        branch_features = self.block1_branch(inputs)
        merged_features = self.concatenate([path_features, branch_features])
        reshaped_features = self.reshape1(merged_features)
        permuted_features = self.permute1(reshaped_features)
        reshaped_features = self.reshape2(permuted_features)
        channel_shuffled_features = self.channel_shuffle(reshaped_features)
        output = self.dense(channel_shuffled_features)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 64, 64, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)