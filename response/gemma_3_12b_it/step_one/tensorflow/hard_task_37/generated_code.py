import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Add, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def create_block():
            x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

            x_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_input)

            x = Add()([x1, x2])
            x = Add()([x, x_conv])

            x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
            return x

        # Define the input layer
        self.input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        x_input = self.input_layer

        # Create the two parallel branches
        branch1_output = create_block()
        branch2_output = create_block()

        # Concatenate the outputs of the two branches
        merged_output = Concatenate(axis=-1)([branch1_output, branch2_output])

        # Flatten the merged output
        flattened_output = Flatten()(merged_output)

        # Add a fully connected layer
        dense_output = Dense(units=10, activation='softmax')(flattened_output)

        self.output_layer = dense_output


    def call(self, inputs):
        return self.output_layer(inputs)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)