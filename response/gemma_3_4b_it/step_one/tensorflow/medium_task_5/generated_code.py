import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))
        ])
        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu')
        ])
        self.max_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2))
        self.addition = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        main_output = self.main_path(inputs)
        branch_output = self.branch_path(inputs)
        combined_output = self.addition([main_output, branch_output])
        pooled_output = self.max_pool(combined_output)
        flattened_output = self.flatten(pooled_output)
        dense1_output = self.dense1(flattened_output)
        output = self.dense2(dense1_output)
        return output


if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data (example - you might need more preprocessing)
    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    # Instantiate the DL_Model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, 32x32 image, 3 color channels
    
    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output tensor shape:", output_tensor.shape)