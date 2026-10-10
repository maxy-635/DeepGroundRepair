import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims='mask'),
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dense(units='mask', activation='relu'),
            tf.keras.layers.Reshape(target_shape=(1024, 1, 1))
        ])

        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1024, activation='relu')
        ])

    def __call__(self, input_tensor):
        main_output = self.main_path(input_tensor)  # output tensor shape of this layer should be (1, 1024, 1, 1)
        branch_output = self.branch_path(input_tensor)  # output tensor shape: (1, 32, 32, 1024)

        combined_output = tf.keras.layers.Add()([main_output, branch_output])
        final_output = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1024, activation='relu'),
            tf.keras.layers.Dense(units=1024, activation='relu')
        ])(combined_output)

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)