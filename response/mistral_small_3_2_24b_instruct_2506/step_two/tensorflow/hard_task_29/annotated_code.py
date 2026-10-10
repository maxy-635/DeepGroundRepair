import tensorflow as tf
from tensorflow.keras.layers import Layer

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.add = tf.math.add

        # Second block: max pooling layers with different scales
        self.max_pool1 = tf.raw_ops.MaxPoolV2(input='mask', ksize='mask', strides='mask', padding='mask')
        self.max_pool2 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.max_pool4 = tf.raw_ops.MaxPoolV2(input=None, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
        self.flatten = tf.keras.layers.Flatten()
        self.concatenate = tf.experimental.numpy.concatenate

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        main_path = self.conv1(inputs)
        main_path = self.conv2(main_path)
        branch_path = inputs
        block1_output = self.add(main_path, branch_path)

        # Second block
        pool1 = self.max_pool1(block1_output)
        pool2 = self.max_pool2(block1_output)
        pool4 = self.max_pool4(block1_output)

        pool1_flat = self.flatten(pool1)
        pool2_flat = self.flatten(pool2)
        pool4_flat = self.flatten(pool4)

        block2_output = self.concatenate([pool1_flat, pool2_flat, pool4_flat], axis=1)

        # Fully connected layers
        output = self.dense1(block2_output)
        output = self.dense2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal([1, 28, 28, 1])  # Dummy input tensor for MNIST (batch_size=1, height=28, width=28, channels=1)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)