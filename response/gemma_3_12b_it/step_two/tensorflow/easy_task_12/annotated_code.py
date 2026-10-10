import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, ReLU, MaxPool2D, Flatten, Dense, add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.block1_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='block1_conv1')
        self.block1_relu1 = ReLU(name='block1_relu1')
        self.block1_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='block1_conv2')
        self.block1_pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool1')

        self.block2_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='block2_conv1')
        self.block2_relu1 = ReLU(name='block2_relu1')
        self.block2_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='block2_conv2')
        self.block2_pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool1')

        # Branch Path
        self.branch_conv1 = Conv2D(filters='mask', kernel_size='mask', strides='mask', padding='mask', activation='relu', name='branch_conv1')

        # Final Layers
        self.flatten = Flatten(name='flatten')
        self.dense = Dense(units=10, activation='softmax', name='dense')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main Path
        main_path = self.block1_conv1(inputs)  # output tensor shape: (1, 28, 28, 1)
        main_path = self.block1_relu1(main_path)  # output tensor shape: (1, 28, 28, 1)
        main_path = self.block1_conv2(main_path)  # output tensor shape: (1, 28, 28, 1)
        main_path = self.block1_pool1(main_path)  # output tensor shape: (1, 14, 14, 1)

        main_path = self.block2_conv1(main_path)  # output tensor shape: (1, 14, 14, 1)
        main_path = self.block2_relu1(main_path)  # output tensor shape: (1, 14, 14, 1)
        main_path = self.block2_conv2(main_path)  # output tensor shape: (1, 14, 14, 1)
        main_path = self.block2_pool1(main_path)  # output tensor shape: (1, 7, 7, 1)

        # Branch Path
        branch_path = self.branch_conv1(inputs)  # output tensor shape of this layer should be (1, 7, 7, 1)

        # Summation
        merged_path = add([main_path, branch_path])

        # Final Layers
        flattened_path = self.flatten(merged_path)
        output = self.dense(flattened_path)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor for MNIST
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)