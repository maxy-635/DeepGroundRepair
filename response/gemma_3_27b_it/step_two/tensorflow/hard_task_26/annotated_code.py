import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch 1
        self.branch1_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 2
        self.branch2_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.branch2_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch2_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))

        # Branch 3
        self.branch3_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.branch3_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch3_upsample = tf.keras.layers.UpSampling2D(size=(2, 2))

        self.main_path_concat = tf.keras.layers.Concatenate(axis=-1)
        self.main_path_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch Path
        self.branch_path_conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Classification Layers
        self.add_layer = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        main_path = self.main_path_conv1(inputs)

        # Branch 1
        branch1 = self.branch1_conv(main_path)

        # Branch 2
        branch2 = self.branch2_pool(main_path)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(main_path)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate Main Path Branches
        main_path = self.main_path_concat([branch1, branch2, branch3])
        main_path = self.main_path_conv2(main_path)

        # Branch Path
        branch_path = self.branch_path_conv1(inputs)

        # Add Main and Branch Paths
        output = self.add_layer([main_path, branch_path])

        # Classification
        output = self.flatten(output)
        output = self.dense1(output)
        output = self.dense2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)