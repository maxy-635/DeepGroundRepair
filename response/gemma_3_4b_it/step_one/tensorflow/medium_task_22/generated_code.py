import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutions
        self.branch1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.branch2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Branch 3: Max pooling
        self.branch3 = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )

        # Multi-scale feature fusion block
        self.merge = tf.keras.layers.Concatenate(axis=-1)

        # Flatten
        self.flatten = tf.keras.layers.Flatten()

        # Fully connected layers
        self.dense1 = tf.keras.layers.Dense(
            units=128,
            activation='relu'
        )
        self.dense2 = tf.keras.layers.Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # Branch 1
        branch1 = self.branch1(inputs)

        # Branch 2
        branch2 = self.branch2(inputs)

        # Branch 3
        branch3 = self.branch3(inputs)

        # Multi-scale feature fusion
        merged = self.merge([branch1, branch2, branch3])

        # Flatten
        flattened = self.flatten(merged)

        # Fully connected layers
        dense1 = self.dense1(flattened)
        output = self.dense2(dense1)

        return output


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data (e.g., normalize)
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the model
    model = DL_Model()

    # Dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    output_tensor = model(input_tensor)

    print("Output shape:", output_tensor.shape)