import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same')
        ])
        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        main_path_output = self.main_path(inputs)  # output tensor shape: (1, 16, 16, 128)
        branch_path_output = self.branch_path(inputs)  # output tensor shape: (1, 1, 1, 10)
        
        reshaped_branch = tf.reshape(branch_path_output, (-1, 10))  # output tensor shape: (1, 10)
        
        combined_output = tf.math.multiply(main_path_output, reshaped_branch)
        
        combined_output = tf.keras.layers.Add()( [combined_output, main_path_output])
        
        final_output = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)(combined_output)
        
        return final_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)