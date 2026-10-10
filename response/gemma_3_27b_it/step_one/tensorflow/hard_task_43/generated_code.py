import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel Average Pooling
        self.avg_pool_1x1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool_2x2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool_4x4 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten_1x1 = tf.keras.layers.Flatten()
        self.flatten_2x2 = tf.keras.layers.Flatten()
        self.flatten_4x4 = tf.keras.layers.Flatten()
        self.concat_block1 = tf.keras.layers.Concatenate(axis=1)

        # Fully connected layer between Block 1 and Block 2
        self.dense_block1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.reshape_block2 = tf.keras.layers.Reshape(target_shape=(1, 1, 64))

        # Block 2: Feature Extraction Branches
        self.branch_1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.branch_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.branch_3 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        ])

        self.concat_block2 = tf.keras.layers.Concatenate(axis=3)

        # Classification Layers
        self.dense_1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        pool_1x1 = self.avg_pool_1x1(inputs)
        pool_2x2 = self.avg_pool_2x2(inputs)
        pool_4x4 = self.avg_pool_4x4(inputs)

        flat_1x1 = self.flatten_1x1(pool_1x1)
        flat_2x2 = self.flatten_2x2(pool_2x2)
        flat_4x4 = self.flatten_4x4(pool_4x4)

        concat_block1_output = self.concat_block1([flat_1x1, flat_2x2, flat_4x4])

        # Between Block 1 and Block 2
        dense_block1_output = self.dense_block1(concat_block1_output)
        reshape_block2_output = self.reshape_block2(dense_block1_output)

        # Block 2
        branch_1_output = self.branch_1(reshape_block2_output)
        branch_2_output = self.branch_2(reshape_block2_output)
        branch_3_output = self.branch_3(reshape_block2_output)

        concat_block2_output = self.concat_block2([branch_1_output, branch_2_output, branch_3_output])

        # Classification
        flat_concat = tf.keras.layers.Flatten()(concat_block2_output)
        dense_1_output = self.dense_1(flat_concat)
        output = self.dense_2(dense_1_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)