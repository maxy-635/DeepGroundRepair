import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.main_path_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch Path
        self.branch_path_gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=False)
        self.branch_path_dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.branch_path_dense2 = tf.keras.layers.Dense(units=3, activation='sigmoid')  # Output 3 channel weights
        self.branch_path_reshape = tf.keras.layers.Reshape(target_shape=(1, 1, 3))

        # Final Layers
        self.final_dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.final_dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        main_path = self.main_path_conv1(inputs)
        main_path = self.main_path_conv2(main_path)
        main_path = self.main_path_conv3(main_path)
        main_path = self.main_path_pool(main_path)

        # Branch Path
        branch_path = self.branch_path_gap(inputs)
        branch_path = self.branch_path_dense1(branch_path)
        branch_path = self.branch_path_dense2(branch_path)
        branch_path = self.branch_path_reshape(branch_path)

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