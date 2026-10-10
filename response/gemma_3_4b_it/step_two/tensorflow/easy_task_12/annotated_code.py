import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')
        ])
        self.branch_path = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')
        ])
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        main_output = self.main_path(inputs)  # output tensor shape of this layer should be (14, 14, 16)
        branch_output = self.branch_path(inputs)
        
        merged = tf.keras.layers.add([main_output, branch_output])
        
        flattened = self.flatten(merged)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)