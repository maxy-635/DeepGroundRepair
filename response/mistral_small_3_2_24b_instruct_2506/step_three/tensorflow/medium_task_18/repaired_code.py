import tensorflow as tf
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution
        self.conv1x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')
        # 3x3 convolution
        self.conv3x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        # 5x5 convolution
        self.conv5x5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')
        # 3x3 max pooling
        self.max_pool = tf.raw_ops.MaxPoolV2(input=inputs, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        # 1x1 convolution after pooling
        self.conv1x1_pool = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()
        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Apply 1x1, 3x3, and 5x5 convolutions
        conv1x1_out = self.conv1x1(inputs)
        conv3x3_out = self.conv3x3(inputs)
        conv5x5_out = self.conv5x5(inputs)

        # Apply 3x3 max pooling and then 1x1 convolution
        pool_out = self.max_pool(inputs)
        conv1x1_pool_out = self.conv1x1_pool(pool_out)

        # Concatenate all feature maps
        concatenated = tf.experimental.numpy.concatenate([conv1x1_out, conv3x3_out, conv5x5_out, conv1x1_pool_out], axis=3)

        # Flatten and pass through fully connected layers
        flattened = self.flatten(concatenated)
        dense1_out = self.dense1(flattened)
        output = self.dense2(dense1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)