import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Max Pooling Layers
        self.max_pool_1x1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.max_pool_2x2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool_4x4 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1 = tf.keras.layers.Flatten()
        self.flatten_2 = tf.keras.layers.Flatten()
        self.flatten_3 = tf.keras.layers.Flatten()
        self.concatenate_block1 = tf.keras.layers.Concatenate(axis=1)

        # Fully connected layer and reshape
        self.dense_block1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.reshape_block2 = tf.keras.layers.Reshape(target_shape=(4, 4, 8))

        # Block 2: Multiple Branches
        self.conv1x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.max_pool_3x3 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.concatenate_block2 = tf.keras.layers.Concatenate(axis=-1)

        # Classification Layers
        self.flatten_block2 = tf.keras.layers.Flatten()
        self.dense_classification = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        pool_1x1 = self.max_pool_1x1(inputs)
        pool_2x2 = self.max_pool_2x2(inputs)
        pool_4x4 = self.max_pool_4x4(inputs)

        flat_1x1 = self.flatten_1(pool_1x1)
        flat_2x2 = self.flatten_2(pool_2x2)
        flat_4x4 = self.flatten_3(pool_4x4)

        concat_block1 = self.concatenate_block1([flat_1x1, flat_2x2, flat_4x4])

        # Fully connected and reshape
        dense_block1 = self.dense_block1(concat_block1)
        reshape_block2 = self.reshape_block2(dense_block1)

        # Block 2
        conv1x1_out = self.conv1x1(reshape_block2)
        conv3x3_out = self.conv3x3(reshape_block2)
        conv5x5_out = self.conv5x5(reshape_block2)
        max_pool_3x3_out = self.max_pool_3x3(reshape_block2)

        concat_block2 = self.concatenate_block2([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_3x3_out])

        # Classification
        flatten_block2 = self.flatten_block2(concat_block2)
        classification = self.dense_classification(flatten_block2)

        return classification


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)