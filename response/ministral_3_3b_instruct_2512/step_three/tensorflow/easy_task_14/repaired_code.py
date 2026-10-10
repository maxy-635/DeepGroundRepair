import tensorflow as tf

class DL_Model:
    def __init__(self):
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.permute = tf.keras.layers.Permute((2, 3, 1))  # Rearrange dimensions to (batch, height, width, channels)
        self.reshape = tf.keras.layers.Reshape(target_shape=(32, 32, 1024))
        self.multiply = tf.keras.layers.Multiply()
        self.flatten = tf.keras.layers.Flatten()
        self.final_dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def __call__(self, input_tensor):
        x = self.global_avg_pool(input_tensor)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.permute(x)  # Rearrange dimensions before reshape
        x = self.reshape(x)
        x = self.multiply([x, input_tensor])
        x = self.flatten(x)
        x = self.final_dense(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)