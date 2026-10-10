import tensorflow as tf
from tensorflow.keras.layers import Lambda, DepthwiseConv2D, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the last dimension
        self.split = Lambda(lambda x: tf.split(x, 3, axis=-1))

        # Define three depthwise separable convolutional layers with different kernel sizes
        self.conv1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Concatenate the outputs of the three groups
        self.concatenate = Concatenate(axis=-1)

        # Flatten the fused features
        self.flatten = Flatten()

        # Fully connected layer for classification
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups
        split_inputs = self.split(inputs)

        # Process each group with depthwise separable convolution
        conv1x1_out = self.conv1x1(split_inputs[0])
        conv3x3_out = self.conv3x3(split_inputs[1])
        conv5x5_out = self.conv5x5(split_inputs[2])

        # Concatenate the outputs
        concatenated = self.concatenate([conv1x1_out, conv3x3_out, conv5x5_out])

        # Flatten and pass through the fully connected layer
        flattened = self.flatten(concatenated)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)