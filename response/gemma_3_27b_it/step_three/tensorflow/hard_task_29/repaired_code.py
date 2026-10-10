import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.add_layer = tf.keras.layers.Add()

        # Block 2: Max Pooling Layers
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()
        self.flatten3 = tf.keras.layers.Flatten()
        self.concatenate_layer = tf.keras.layers.Concatenate(axis=-1)

        # Fully Connected Layers
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Block 1
        main_path = self.conv1(inputs)  # output tensor shape: (1, 28, 28, 32)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 28, 28, 1)
        branch_path = inputs  # output tensor shape: (1, 28, 28, 1)
        output_block1 = self.add_layer([main_path, branch_path])  # output tensor shape: (1, 28, 28, 1)

        # Block 2
        pool1 = self.maxpool1(inputs)  # output tensor shape: (1, 28, 28, 1)
        pool2 = self.maxpool2(inputs)  # output tensor shape: (1, 14, 14, 1)
        pool3 = self.maxpool3(inputs)  # output tensor shape: (1, 7, 7, 1)
        flat1 = self.flatten1(pool1)  # output tensor shape: (1, 784)
        flat2 = self.flatten2(pool2)  # output tensor shape: (1, 196)
        flat3 = self.flatten3(pool3)  # output tensor shape: (1, 49)
        output_block2 = self.concatenate_layer([flat1, flat2, flat3])  # output tensor shape: (1, 1029)

        # Combine and pass through fully connected layers
        combined = self.concatenate_layer([output_block1, output_block2])
        x = self.dense1(combined)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)