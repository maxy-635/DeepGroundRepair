import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def create_block(input_tensor):
            x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
            x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
            x = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
            x = Add()([x, input_tensor])
            return x

        # Three branches
        branch1 = create_block(tf.keras.Input(shape=(28, 28, 1)))
        branch2 = create_block(branch1)
        branch3 = create_block(branch2)

        # Concatenate the outputs of the three branches
        concatenated = Concatenate()([branch1, branch2, branch3])

        # Flatten the concatenated tensor
        flattened = Flatten()(concatenated)

        # Fully connected layer for classification
        output = Dense(units=10, activation='softmax')(flattened)

        self.build(input_shape=(None, 28, 28, 1))
        self.output_layer = output


    def call(self, inputs):
        return self.output_layer(inputs)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)