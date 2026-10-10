import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1x1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2x2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool4x4 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.flatten2 = tf.keras.layers.Flatten()
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.flatten3 = tf.keras.layers.Flatten()
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # Between Block 1 and Block 2
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.reshape1 = tf.keras.layers.Reshape(target_shape='mask')

        # Block 2
        self.conv1x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool_avg = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv1x1_avg = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        # Final Classification Layers
        self.dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        x1 = self.pool1x1(inputs)  # output tensor shape: (1, 28, 28, 1)
        x1 = self.flatten1(x1)  # output tensor shape: (1, 784)
        x1 = self.dropout1(x1)  # output tensor shape: (1, 784)

        x2 = self.pool2x2(inputs)  # output tensor shape: (1, 14, 14, 1)
        x2 = self.flatten2(x2)  # output tensor shape: (1, 196)
        x2 = self.dropout2(x2)  # output tensor shape: (1, 196)

        x3 = self.pool4x4(inputs)  # output tensor shape: (1, 7, 7, 1)
        x3 = self.flatten3(x3)  # output tensor shape: (1, 49)
        x3 = self.dropout3(x3)  # output tensor shape: (1, 49)

        x = self.concat1([x1, x2, x3])  # output tensor shape: (1, 1029)

        # Between Block 1 and Block 2
        x = self.dense1(x)  # output tensor shape: (1, 64)
        x = self.reshape1(x)

        # Block 2
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3_1(x)
        branch3 = self.conv3x3_2(x)
        branch4 = self.conv3x3_3(x)
        branch5 = self.pool_avg(x)
        branch5 = self.conv1x1_avg(branch5)

        x = self.concat2([branch1, branch2, branch3, branch4, branch5])

        # Final Classification Layers
        x = self.dense2(x)
        x = self.dropout4(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)