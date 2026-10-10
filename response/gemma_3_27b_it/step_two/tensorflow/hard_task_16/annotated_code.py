import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_group1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)
        self.conv1_group2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)
        self.conv1_group3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)

        # Transition Convolution
        self.transition_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)

        # Block 2
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D(keepdims=False)
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=32, activation=None) # No activation here
        self.reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 32))
        

        # Branch
        self.branch = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)

        # Final Classification
        self.add_layer = tf.keras.layers.Add()
        self.dense_final = tf.keras.layers.Dense(units=10, activation='softmax')
        

    def call(self, inputs):
        # Block 1
        group1 = self.conv1_group1(inputs)
        group2 = self.conv1_group2(inputs)
        group3 = self.conv1_group3(inputs)
        block1_output = tf.keras.layers.Concatenate(axis=-1)([group1, group2, group3])

        # Transition Convolution
        adjusted_output = self.transition_conv(block1_output)

        # Block 2
        pooled_output = self.global_max_pool(adjusted_output)
        channel_weights = self.dense1(pooled_output)
        channel_weights = self.dense2(channel_weights)
        channel_weights = self.reshape(channel_weights)
        main_path_output = tf.keras.layers.Multiply()([adjusted_output, channel_weights])

        # Branch
        branch_output = self.branch(inputs)

        # Add Main Path and Branch
        combined_output = self.add_layer([main_path_output, branch_output])

        # Final Classification
        final_output = self.dense_final(combined_output)

        return final_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)