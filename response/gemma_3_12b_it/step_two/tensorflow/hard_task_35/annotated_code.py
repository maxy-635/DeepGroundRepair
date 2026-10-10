import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def build_block(input_tensor):
            # Global average pooling
            pooled = GlobalAveragePooling2D(keepdims=False)(input_tensor)

            # Fully connected layers
            dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pooled)
            dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)

            # Reshape to match input shape
            weights = tf.reshape(dense2, shape=input_tensor.shape)

            # Element-wise multiplication
            weighted_output = tf.math.multiply(input_tensor, weights)
            return weighted_output

        # Two branches with the same block
        self.branch1 = tf.keras.Sequential([
            build_block
        ])
        self.branch2 = tf.keras.Sequential([
            build_block
        ])

        # Concatenate the outputs of the two branches
        self.concat = Concatenate(axis=-1)

        # Flatten and final fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')  # Assuming 10 classes for CIFAR-10

    def call(self, inputs):
        # Pass inputs through both branches
        output1 = self.branch1(inputs)
        output2 = self.branch2(inputs)

        # Concatenate the outputs
        concatenated = self.concat([output1, output2])

        # Flatten and pass through the final dense layer
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)