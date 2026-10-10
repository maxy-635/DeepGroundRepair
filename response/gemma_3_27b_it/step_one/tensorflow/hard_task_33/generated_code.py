import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def block(x, filters):
            # Elevate dimension
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
            # Depthwise separable convolution
            x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
            # Reduce dimension
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
            # Add input
            x = tf.keras.layers.Add()([x, tf.identity(x)])  # Use tf.identity to get the original input
            return x

        # Three branches
        input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
        branch1 = block(input_layer, 32)
        branch2 = block(input_layer, 64)
        branch3 = block(input_layer, 128)

        # Concatenate branches
        merged = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2, branch3])

        # Flatten and fully connected layer
        flatten = tf.keras.layers.Flatten()(merged)
        output = tf.keras.layers.Dense(units=10, activation='softmax')(flatten)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output)

    def call(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = tf.random.normal((1, 28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)