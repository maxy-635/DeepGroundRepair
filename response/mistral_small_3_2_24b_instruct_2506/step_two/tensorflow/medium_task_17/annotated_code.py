import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)  # output tensor shape: (4)
        # Reshape the input tensor into three groups
        height = input_shape[1]  # output tensor shape: (1)
        width = input_shape[2]  # output tensor shape: (1)
        channels = input_shape[3]  # output tensor shape: (1)
        groups = 3  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)
        reshaped = tf.reshape(inputs, (height, width, groups, channels_per_group))  # output tensor shape: (32, 32, 3, 1)
        # Swap the third and fourth dimensions
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        # Reshape back to the original input shape
        reshaped_back = tf.reshape(permuted, (height, width, channels))
        # Pass through a fully connected layer with softmax activation
        output = self.dense(reshaped_back)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)