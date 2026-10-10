import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, Conv2D, concatenate, Dense, Flatten, multiply, Reshape, UpSampling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the two pathways with repeated block structure
        self.pathway1 = self._build_pathway()
        self.pathway2 = self._build_pathway()

    def _build_pathway(self):
        # Define a single block with batch normalization, ReLU, and 3x3 convolution
        block = tf.keras.Sequential([
            BatchNormalization(axis=-1),
            ReLU(),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        ])
        # Create a repeated block structure executed three times
        repeated_block = tf.keras.Sequential([
            block,
            block,
            block
        ])
        return repeated_block

    def call(self, inputs):
        # Process inputs through both pathways
        pathway1_output = self.pathway1(inputs)
        pathway2_output = self.pathway2(inputs)

        # Concatenate outputs along the channel dimension
        concatenated_output = concatenate([pathway1_output, pathway2_output], axis=-1)

        # Flatten and classify using two fully connected layers
        flattened_output = Flatten()(concatenated_output)
        dense1 = Dense(units=128, activation='relu')(flattened_output)
        output = Dense(units=10, activation='softmax')(dense1)

        return output

if __name__ == "__main__":
    # Instantiate the model and provide a dummy input tensor
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input shape for MNIST (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)