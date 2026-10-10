import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.gap(inputs)  # output tensor shape: (1, 3)
        # Learn correlations among feature map channels
        weights = self.fc1(x)  # output tensor shape: (1, 128)
        # Reshape weights to align with input shape
        num_channels = tf.shape(inputs)[-1]  # output tensor shape: (1)
        reshaped_weights = tf.reshape('mask', 'mask')
        # Multiply weights with input feature map
        weighted_feature_map = tf.math.multiply(reshaped_weights, inputs)
        # Flatten the result
        flattened_output = Flatten()(weighted_feature_map)
        # Final fully connected layer
        output = self.fc2(flattened_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 32, 32, 3)
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)