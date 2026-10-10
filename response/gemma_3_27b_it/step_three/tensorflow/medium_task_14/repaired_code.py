import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        block1_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        block1_bn = tf.keras.layers.BatchNormalization(axis=-1)
        block1_relu = tf.keras.layers.ReLU()
        self.block1 = tf.keras.Sequential([block1_conv, block1_bn, block1_relu])

        # Block 2
        block2_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        block2_bn = tf.keras.layers.BatchNormalization(axis=-1)
        block2_relu = tf.keras.layers.ReLU()
        self.block2 = tf.keras.Sequential([block2_conv, block2_bn, block2_relu])

        # Block 3
        block3_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        block3_bn = tf.keras.layers.BatchNormalization(axis=-1)
        block3_relu = tf.keras.layers.ReLU()
        self.block3 = tf.keras.Sequential([block3_conv, block3_bn, block3_relu])
        
        # Parallel Branch
        parallel_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        parallel_bn = tf.keras.layers.BatchNormalization(axis=-1)
        parallel_relu = tf.keras.layers.ReLU()
        self.parallel_branch = tf.keras.Sequential([parallel_conv, parallel_bn, parallel_relu])

        # Fully Connected Layers
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        # Process through blocks
        block1_output = self.block1(inputs)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)

        # Parallel Branch
        parallel_output = self.parallel_branch(inputs)

        # Add outputs
        added_output = tf.keras.layers.Add()([block3_output, parallel_output])

        # Flatten and pass through fully connected layers
        flattened_output = tf.keras.layers.Flatten()(added_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)