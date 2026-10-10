import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Pathway 1
        self.pathway1_block1 = self.create_block()
        self.pathway1_block2 = self.create_block()
        self.pathway1_block3 = self.create_block()

        # Pathway 2
        self.pathway2_block1 = self.create_block()
        self.pathway2_block2 = self.create_block()
        self.pathway2_block3 = self.create_block()

        # Merge and Classify
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def create_block(self):
        return tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        ])

    def call(self, inputs):
        # Pathway 1
        pathway1 = inputs
        for block in [self.pathway1_block1, self.pathway1_block2, self.pathway1_block3]:
            pathway1 = self.concatenate([pathway1, block(pathway1)])

        # Pathway 2
        pathway2 = inputs
        for block in [self.pathway2_block1, self.pathway2_block2, self.pathway2_block3]:
            pathway2 = self.concatenate([pathway2, block(pathway2)])

        # Merge
        merged = self.concatenate([pathway1, pathway2])

        # Classify
        x = self.dense1(merged)
        return self.dense2(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input tensor for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)