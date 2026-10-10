import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, Concatenate, Add, Flatten, Dense
from tensorflow.keras import Sequential

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block
        self.main_path_block = Sequential([
            DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu', name='depthwise_conv1'),
            ReLU(name='relu1'),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None, name='conv1')
        ])

        # Main path
        self.main_path = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='main_conv1'),
            self.main_path_block,
            self.main_path_block,
            self.main_path_block
        ])

        # Branch path
        self.branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='branch_conv1')

        # Fusion
        self.fusion = Add(name='fusion')

        # Classifier
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(units=10, activation='softmax', name='dense')

    def call(self, inputs):
        # Main path
        main_path_output = self.main_path(inputs)

        # Branch path
        branch_path_output = self.branch_path(inputs)

        # Fusion
        fused_output = self.fusion([main_path_output, branch_path_output])

        # Classifier
        flattened_output = self.flatten(fused_output)
        output = self.dense(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # MNIST input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)