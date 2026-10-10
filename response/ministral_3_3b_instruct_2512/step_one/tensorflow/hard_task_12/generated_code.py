class DL_Model:
    def __init__(self):
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation=None)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, dummy_kernel_size, dummy_strides, dummy_padding, dummy_activation)
        self.concatenate = tf.keras.layers.Concatenate(axis=1)
        self.add = tf.keras.layers.Add()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10, activation='softmax')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        y = self.conv2(x)
        z = self.conv3(x)
        concatenated = self.concatenate([y, z])
        branch_output = self.conv4(input_tensor)
        combined = self.add([concatenated, branch_output])
        flattened = self.flatten(combined)
        output = self.dense1(flattened)
        output = self.dense2(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)