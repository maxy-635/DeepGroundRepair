import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Average Pooling and Flattening
        self.avg_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.avg_pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.avg_pool3 = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, 64))

        # Block 2: Multi-Scale Feature Extraction
        # Path 1
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)

        # Path 2
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)

        # Path 3
        self.conv3_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(rate=0.2)

        # Path 4
        self.avg_pool4 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv4_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.dropout4 = tf.keras.layers.Dropout(rate=0.2)

        self.concat_paths = tf.keras.layers.Concatenate(axis=-1)
        self.dense2 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=10, activation='softmax')  # Assuming 10 classes for MNIST

    def call(self, inputs):
        # Block 1
        x1 = self.avg_pool1(inputs)
        x2 = self.avg_pool2(inputs)
        x3 = self.avg_pool3(inputs)

        x1_flat = self.flatten(x1)
        x2_flat = self.flatten(x2)
        x3_flat = self.flatten(x3)

        x_concat = self.concat([x1_flat, x2_flat, x3_flat])
        x = self.dense1(x_concat)
        x = self.reshape(x)

        # Block 2
        path1 = self.conv1_1(x)
        path1 = self.dropout1(path1)

        path2 = self.conv2_1(x)
        path2 = self.conv2_2(path2)
        path2 = self.conv2_3(path2)
        path2 = self.dropout2(path2)

        path3 = self.conv3_1(x)
        path3 = self.conv3_2(path3)
        path3 = self.dropout3(path3)

        path4 = self.avg_pool4(x)
        path4 = self.conv4_1(path4)
        path4 = self.dropout4(path4)

        x_paths = self.concat_paths([path1, path2, path3, path4])
        x = self.dense2(x_paths)
        return self.dense3(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input tensor for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)