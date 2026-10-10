import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block with global average pooling and two dense layers
        self.block = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Dense(units=32, activation='relu'),  # Assuming input channels are 32 for CIFAR-10
            layers.Dense(units=32, activation='relu')
        ])

    def call(self, inputs):
        # Branch 1: Apply block to inputs
        branch1 = self.block(inputs)

        # Reshape the output to match input spatial dimensions (32, 32)
        branch1 = layers.Reshape(target_shape=(32, 32))(branch1)

        # Branch 2: Apply block to inputs again
        branch2 = self.block(inputs)

        # Reshape the output to match input spatial dimensions (32, 32)
        branch2 = layers.Reshape(target_shape=(32, 32))(branch2)

        # Element-wise multiplication of both branches
        output = layers.Multiply()([branch1, branch2])

        # Concatenate along the channel dimension
        output = layers.Concatenate(axis=3)(output)

        # Flatten and add final dense layer
        output = layers.Flatten()(output)
        output = layers.Dense(units=10, activation='softmax')(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)