import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.dropout = tf.keras.layers.Dropout(rate=0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main pathway
        x1 = self.conv1(inputs)
        x1 = self.dropout(x1)
        x2 = self.conv2(x1)
        x2 = self.dropout(x2)
        x3 = self.conv3(x2)
        x3 = self.dropout(x3)
        main_pathway = x3

        # Parallel pathway
        parallel_pathway = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(inputs)
        parallel_pathway = self.flatten(parallel_pathway)

        # Concatenate pathways
        merged = tf.keras.layers.Concatenate(axis=-1)([main_pathway, parallel_pathway])

        # Dense layers
        merged = self.dense1(merged)
        output = self.dense2(merged)

        return output


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data (example: rescale to [0, 1])
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the model
    model = DL_Model()

    # Dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)

    print("Output shape:", output_tensor.shape)