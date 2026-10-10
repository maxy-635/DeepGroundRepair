import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Splitting into three channel groups (assuming 3 channels)
        # Each group will have separate feature extraction paths

        # Group 1
        self.conv1_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_5 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Group 2
        self.conv2_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_5 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Group 3
        self.conv3_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3_5 = tf.keras.layers.DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs from the three groups
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')  # 10 classes in CIFAR-10

    def call(self, inputs):
        # Split the input into three channel groups
        r, g, b = tf.split(inputs, num_or_size_splits=3, axis=-1)

        # Feature extraction for each group
        group1 = self.conv1_1(r)
        group1 = self.conv1_3(group1)
        group1 = self.conv1_5(group1)

        group2 = self.conv2_1(g)
        group2 = self.conv2_3(group2)
        group2 = self.conv2_5(group2)

        group3 = self.conv3_1(b)
        group3 = self.conv3_3(group3)
        group3 = self.conv3_5(group3)

        # Concatenate the features
        concatenated = self.concatenate([group1, group2, group3])

        # Fully connected layers
        flattened = self.flatten(concatenated)
        dense1 = self.dense1(flattened)
        dense2 = self.dense2(dense1)
        output = self.dense3(dense2)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with the expected shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)