import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        branch1_output = self.branch1(inputs)
        branch2_output = self.branch2(inputs)

        concatenated = tf.keras.layers.Concatenate(axis=-1)([branch1_output, branch2_output])
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)