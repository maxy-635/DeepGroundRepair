import tensorflow as tf
from tensorflow.keras import layers

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.reshape_layer = layers.Reshape(target_shape=(32, 32, 3, 3))
        self.permute_layer = layers.Permute(dims=(2, 3))
        self.reshape_back_layer = layers.Reshape(target_shape=(32, 32, 9))
        self.dense_layer = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Reshape the input tensor to (height, width, groups, channels_per_group)
        reshaped_tensor = self.reshape_layer(inputs)
        # Swap the third and fourth dimensions for channel shuffling
        permuted_tensor = self.permute_layer(reshaped_tensor)
        # Reshape back to original input shape
        reshaped_back_tensor = self.reshape_back_layer(permuted_tensor)
        # Pass through a fully connected layer with softmax activation
        output = self.dense_layer(reshaped_back_tensor)
        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)