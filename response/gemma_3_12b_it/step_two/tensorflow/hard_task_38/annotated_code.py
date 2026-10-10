import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Pathway 1
        self.conv_block_1_pathway_1 = self.create_conv_block()
        self.conv_block_2_pathway_1 = self.create_conv_block()
        self.conv_block_3_pathway_1 = self.create_conv_block()

        # Pathway 2
        self.conv_block_1_pathway_2 = self.create_conv_block()
        self.conv_block_2_pathway_2 = self.create_conv_block()
        self.conv_block_3_pathway_2 = self.create_conv_block()

        # Merge and Classify
        self.concat = Concatenate(axis=-1)
        self.dense_1 = Dense(128, activation='relu')
        self.dense_2 = Dense(10, activation='softmax')

    def create_conv_block(self):
        return tf.keras.Sequential([
            BatchNormalization(axis=-1),
            ReLU(),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        ])

    def call(self, inputs):
        # Pathway 1
        x = self.conv_block_1_pathway_1(inputs)
        x = self.conv_block_2_pathway_1(x)
        x = self.conv_block_3_pathway_1(x)

        # Pathway 2
        y = self.conv_block_1_pathway_2(inputs)
        y = self.conv_block_2_pathway_2(y)
        y = self.conv_block_3_pathway_2(y)

        # Concatenate outputs from both pathways
        merged = self.concat([x, y])

        # Classification
        x = self.dense_1(merged)
        output = self.dense_2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Example input tensor for MNIST
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)