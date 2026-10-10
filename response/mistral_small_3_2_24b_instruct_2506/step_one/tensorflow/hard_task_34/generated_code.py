import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, ReLU, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path blocks
        self.main_conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.main_relu1 = ReLU()
        self.main_concat1 = Concatenate(axis=-1)

        self.main_conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.main_relu2 = ReLU()
        self.main_concat2 = Concatenate(axis=-1)

        self.main_conv3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.main_relu3 = ReLU()
        self.main_concat3 = Concatenate(axis=-1)

        # Branch path
        self.branch_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')

        # Fusion and output
        self.add = Add()
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = inputs
        for _ in range(3):
            conv_out = self.main_conv1(x)
            relu_out = self.main_relu1(conv_out)
            x = self.main_concat1([x, relu_out])

        # Branch path
        branch_out = self.branch_conv(inputs)

        # Fusion
        fused_out = self.add([x, branch_out])

        # Output
        flattened = self.flatten(fused_out)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)