import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_group1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_group2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_group3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split into three groups along the channel dimension
        group1 = inputs[:, :, :, :3]
        group2 = inputs[:, :, :, 3:6]
        group3 = inputs[:, :, :, 6:]

        # Apply 1x1 convolutions to each group
        group1 = self.conv1_group1(group1)
        group2 = self.conv1_group2(group2)
        group3 = self.conv1_group3(group3)

        # Downsample each group
        group1 = self.avgpool(group1)
        group2 = self.avgpool(group2)
        group3 = self.avgpool(group3)

        # Concatenate the groups
        concatenated = self.concat([group1, group2, group3])

        # Flatten and pass through fully connected layers
        flattened = self.flatten(concatenated)
        dense1 = self.dense1(flattened)
        output = self.dense2(dense1)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 9))  # Example input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)