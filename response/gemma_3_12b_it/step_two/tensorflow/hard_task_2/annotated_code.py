import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional blocks for each group
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layer for classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Split the input into three groups along the channel dimension
        split_values = tf.split(inputs, num_or_size_splits=3, axis=-1)  # output tensor shape: (unknown)

        # Process each group through the convolutional blocks
        group1 = self.conv1_1(split_values[0])  # output tensor shape: (1, 32, 32, 32)
        group1 = self.conv1_3(group1)  # output tensor shape: (1, 32, 32, 32)
        group1 = self.conv1_2(group1)  # output tensor shape: (1, 32, 32, 32)

        group2 = self.conv2_1(split_values[1])  # output tensor shape: (1, 32, 32, 32)
        group2 = self.conv2_3(group2)  # output tensor shape: (1, 32, 32, 32)
        group2 = self.conv2_2(group2)  # output tensor shape: (1, 32, 32, 32)

        group3 = self.conv3_1(split_values[2])  # output tensor shape: (1, 32, 32, 32)
        group3 = self.conv3_3(group3)  # output tensor shape: (1, 32, 32, 32)
        group3 = self.conv3_2(group3)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Combine the outputs from the three groups
        main_path = tf.add(group1, group2)
        main_path = tf.add(main_path, group3)

        # Fuse the main path with the original input
        combined_features = tf.add(main_path, inputs)

        # Flatten the combined features and feed them into the fully connected layer
        flattened_features = self.flatten(combined_features)
        output = self.dense(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 32, 32, 3) for CIFAR-10
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)