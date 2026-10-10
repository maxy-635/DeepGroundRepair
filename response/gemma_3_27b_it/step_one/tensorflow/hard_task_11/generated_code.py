import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv1x1_main = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)

        # Parallel Branch
        self.conv1x1_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)
        self.conv1x3_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=1, padding='same', activation=tf.keras.activations.relu)
        self.conv3x1_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=1, padding='same', activation=tf.keras.activations.relu)

        # Concatenate and Final Convolution
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.conv1x1_final = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.keras.activations.relu)

        # Additive Connection
        self.add_layer = tf.keras.layers.Add()

        # Classification Layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main Pathway
        main_path = self.conv1x1_main(inputs)

        # Parallel Branch
        branch1 = self.conv1x1_branch(inputs)
        branch2 = self.conv1x3_branch(branch1)
        branch3 = self.conv3x1_branch(branch2)

        # Concatenate
        merged_branch = self.concatenate([branch3, main_path])

        # Final Convolution
        output = self.conv1x1_final(merged_branch)

        # Additive Connection
        output = self.add_layer([output, inputs])

        # Classification
        x = self.flatten(output)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)