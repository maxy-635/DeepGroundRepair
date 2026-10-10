import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1_1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool1_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool1_4 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.concatenate1 = tf.keras.layers.Concatenate(axis=1)

        # Between Block 1 and Block 2
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.reshape1 = tf.keras.layers.Reshape(target_shape=(4, 4, 8))  # Assuming input image size is 28x28

        # Block 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='valid', activation='relu')
        self.conv2_3_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='valid', activation='relu')
        self.pool2_1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.conv2_4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.concatenate2 = tf.keras.layers.Concatenate(axis=-1)

        # Final Classification Layers
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        pool1_1_out = self.pool1_1(inputs)
        pool1_2_out = self.pool1_2(inputs)
        pool1_4_out = self.pool1_4(inputs)

        flatten1_1 = self.flatten1(pool1_1_out)
        flatten1_2 = self.flatten1(pool1_2_out)
        flatten1_4 = self.flatten1(pool1_4_out)

        dropout1_1 = self.dropout1(flatten1_1)
        dropout1_2 = self.dropout1(flatten1_2)
        dropout1_4 = self.dropout1(flatten1_4)

        block1_output = self.concatenate1([dropout1_1, dropout1_2, dropout1_4])

        # Between Block 1 and Block 2
        dense1_out = self.dense1(block1_output)
        reshape1_out = self.reshape1(dense1_out)

        # Block 2
        conv2_1_out = self.conv2_1(reshape1_out)
        conv2_2_1_out = self.conv2_2_1(reshape1_out)
        conv2_2_2_out = self.conv2_2_2(conv2_2_1_out)
        conv2_3_1_out = self.conv2_3_1(reshape1_out)
        conv2_3_2_out = self.conv2_3_2(conv2_3_1_out)
        pool2_1_out = self.pool2_1(reshape1_out)
        conv2_4_out = self.conv2_4(pool2_1_out)

        block2_output = self.concatenate2([conv2_1_out, conv2_2_2_out, conv2_3_2_out, conv2_4_out])

        # Final Classification Layers
        dense2_out = self.dense2(block2_output)
        dropout2_out = self.dropout2(dense2_out)
        output = self.dense3(dropout2_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)