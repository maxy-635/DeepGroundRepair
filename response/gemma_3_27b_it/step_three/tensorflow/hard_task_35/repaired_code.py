import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def block(input_tensor):
            x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(input_tensor)
            x = tf.keras.layers.Dense(units=input_tensor.shape[-1], activation='relu')(x)
            x = tf.keras.layers.Dense(units=input_tensor.shape[-1], activation='sigmoid')(x)
            x = tf.keras.layers.Reshape(target_shape=(1, 1, input_tensor.shape[-1]))(x)
            x = tf.keras.layers.Multiply()([input_tensor, x])
            return x

        # Define the two branches
        input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
        branch1 = block(input_layer)
        branch2 = block(input_layer)

        # Concatenate the outputs of the two branches
        concat_layer = tf.keras.layers.Concatenate(axis=-1)([branch1, branch2])

        # Flatten and apply the final fully connected layer
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(flatten_layer)

        # Define the model
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def call(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)