import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch 1
        self.branch1_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')

        # Branch 2
        self.branch2_avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
        self.branch2_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch2_transposeconv = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='same', activation='relu')

        # Branch 3
        self.branch3_avgpool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')
        self.branch3_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.branch3_transposeconv = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=4, padding='same', activation='relu')

        # Concatenate and Main Path Output
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.main_path_conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Branch Path
        self.branch_path_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu')

        # Fusion
        self.add_layer = tf.keras.layers.Add()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Path
        main_path = self.main_path_conv1(inputs)
        branch1 = self.branch1_conv(main_path)

        branch2 = self.branch2_avgpool(main_path)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_transposeconv(branch2)

        branch3 = self.branch3_avgpool(main_path)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_transposeconv(branch3)

        main_path_features = self.concat([branch1, branch2, branch3])
        main_path_output = self.main_path_conv2(main_path_features)

        # Branch Path
        branch_path = self.branch_path_conv(inputs)

        # Fusion
        fused = self.add_layer([main_path_output, branch_path])

        # Classification
        flattened = self.flatten(fused)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)