import tensorflow as tf
from tensorflow.keras.layers import Lambda, DepthwiseConv2D, Conv2D, Concatenate, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define layers
        self.split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))
        self.depthwise_conv1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concat_layer = Concatenate(axis=-1)
        self.flatten_layer = Flatten()
        self.dense_layer = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Split the input tensor
        split_tensors = self.split_layer(inputs)

        # Apply depthwise separable convolutions to each split
        output1 = self.depthwise_conv1x1(split_tensors[0])
        output2 = self.depthwise_conv3x3(split_tensors[1])
        output3 = self.depthwise_conv5x5(split_tensors[2])

        # Concatenate the outputs
        concatenated_output = self.concat_layer([output1, output2, output3])

        # Flatten the concatenated output
        flattened_output = self.flatten_layer(concatenated_output)

        # Pass the flattened output through the dense layer
        output = self.dense_layer(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)