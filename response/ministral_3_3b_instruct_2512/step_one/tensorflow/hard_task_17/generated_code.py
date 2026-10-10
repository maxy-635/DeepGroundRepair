import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu')
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])
        self.addition = tf.keras.layers.Add()
        self.final_dense = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])

    def call(self, inputs):
        block1_output = self.block1(inputs)
        block2_output = self.block2(inputs)
        branch_output = self.addition([block1_output, block2_output])
        output = self.final_dense(branch_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)