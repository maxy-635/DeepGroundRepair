import tensorflow as tf
from tensorflow.keras.layers import Conv1D, SeparableConv2D, MaxPool2D, Conv2D, Add, Flatten, Dense, ReLU, Conv2DTranspose, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path_block1 = tf.keras.Sequential([
            ReLU(),
            SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)
        ])
        self.main_path_block2 = tf.keras.Sequential([
            ReLU(),
            SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        ])

        # Branch path
        self.branch_path = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size='mask', strides='mask', padding='mask', activation=None)
        ])

        # Summation and final layers
        self.add_layer = Add()
        self.flatten_layer = Flatten()
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        main_path_output = self.main_path_block1(inputs)  # output tensor shape: (1, 28, 28, 32)
        main_path_output = self.main_path_block2(main_path_output)  # output tensor shape: (1, 14, 14, 64)
        branch_path_output = self.branch_path(inputs)  # output tensor shape of this layer should be (1, 14, 14, 64)

        summed_output = self.add_layer([main_path_output, branch_path_output])
        flattened_output = self.flatten_layer(summed_output)
        output = self.final_dense(flattened_output)

        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)