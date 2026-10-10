import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_conv3 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_pool = tf.keras.layers.MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Branch Path
        self.branch_path_gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.branch_path_dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.branch_path_dense2 = tf.keras.layers.Dense(units=3, activation='sigmoid')  # Output 3 channel weights
        self.branch_path_reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 3))

        # Final Layers
        self.final_dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.final_dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main Path
        main_path = self.main_path_conv1(inputs)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.main_path_conv2(main_path)  # output tensor shape: (1, 32, 32, 64)
        main_path = self.main_path_conv3(main_path)  # output tensor shape of this layer should be (1, 32, 32, 3)
        main_path = self.main_path_pool(main_path)  # output tensor shape of this layer should be (1, 32, 32, 128)

        # Branch Path
        branch_path = self.branch_path_gap(inputs)  # output tensor shape: (1, 3)
        branch_path = self.branch_path_dense1(branch_path)  # output tensor shape: (1, 64)
        branch_path = self.branch_path_dense2(branch_path)  # output tensor shape: (1, 3)
        branch_path = self.branch_path_reshape(branch_path)  # output tensor shape: (1, 1, 1, 3)

        # Combine Paths
        combined = tf.keras.layers.Multiply()([main_path, branch_path])
        combined = tf.keras.layers.Add()([main_path, combined])

        # Final Classification
        combined = self.final_dense1(combined)
        outputs = self.final_dense2(combined)

        return outputs


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)