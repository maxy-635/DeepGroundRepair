import tensorflow as tf
from tensorflow.keras.layers import Lambda, DepthwiseConv2D, Conv2D, MaxPool2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Split input into three groups with different kernel sizes
        self.split = Lambda(lambda x: tf.split(x, 3, axis=-1))
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.concat1 = Concatenate(axis=-1)

        # Second block: Multiple branches for feature extraction
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch4_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        split_inputs = self.split(inputs)
        conv1 = self.depthwise_conv1(split_inputs[0])
        conv2 = self.depthwise_conv2(split_inputs[1])
        conv3 = self.depthwise_conv3(split_inputs[2])
        concat1 = self.concat1([conv1, conv2, conv3])

        # Second block
        branch1 = self.branch1(concat1)
        branch2 = self.branch2_conv2(self.branch2_conv1(concat1))
        branch3 = self.branch3_conv2(self.branch3_conv1(concat1))
        branch4 = self.branch4_conv(self.branch4_pool(concat1))
        concat2 = self.concat2([branch1, branch2, branch3, branch4])

        # Final layers
        flattened = self.flatten(concat2)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print (None, 10)