import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the input layer
        self.input_layer = tf.keras.Input(shape=(28, 28, 1))

        # First parallel branch
        self.conv1_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Second parallel branch (same as the first)
        self.conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Parallel branch connecting input directly
        self.conv_input = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Addition layers
        self.add1 = tf.keras.layers.Add()
        self.add2 = tf.keras.layers.Add()

        # Concatenation layer
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Dense layer
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First branch
        x1_1 = self.conv1_1(inputs)
        x1_2 = self.conv1_2(x1_1)
        x1_3 = self.conv1_3(x1_2)

        # Second branch
        x2_1 = self.conv2_1(inputs)
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)

        # Input branch
        x_input = self.conv_input(inputs)
        
        # Addition
        x1 = self.add1([x1_3, x_input])
        x2 = self.add2([x2_3, x_input])
        

        # Concatenation
        x = self.concatenate([x1, x2])

        # Flatten
        x = self.flatten(x)

        # Dense layer
        output = self.dense(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)