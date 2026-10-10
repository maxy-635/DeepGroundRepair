import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three branches
        def create_branch():
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                depth_multiplier=1,
                activation='relu'
            )
            x = tf.keras.layers.Dropout(rate=0.2)
            x = tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation='relu'
            )
            x = tf.keras.layers.Dropout(rate=0.2)
            return x

        branch1 = create_branch()
        branch2 = create_branch()
        branch3 = create_branch()

        # Define the main model flow
        self.branch1 = branch1
        self.branch2 = branch2
        self.branch3 = branch3

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)

        x = self.concatenate([x1, x2, x3])
        x = self.dense1(x)
        x = self.dense2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel (grayscale)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)