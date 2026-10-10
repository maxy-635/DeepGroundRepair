import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense, Conv1D, Lambda

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')
        self.conv_main_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv_main_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.max_pool_main = MaxPool2D(pool_size='mask', strides='mask', padding='mask')
        self.dropout_main = Dropout(rate=0.5)

        # Branch pathway
        self.conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')

        # Fusion layer
        self.add_layer = Add()

        # Final layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 28, 28, 1)
        # Main pathway
        x_main = self.conv_main(inputs)  # output tensor shape: (1, 26, 26, 32)
        x_main = self.conv_main_1x1_1(x_main)  # output tensor shape: (1, 26, 26, 32)
        x_main = self.conv_main_1x1_2(x_main)  # output tensor shape: (1, 26, 26, 32)
        x_main = self.max_pool_main(x_main)  # output tensor shape of this layer should be (1, 11, 11, 32)
        x_main = self.dropout_main(x_main)

        # Branch pathway
        x_branch = self.conv_branch(x_main)  # output tensor shape: (1, 11, 11, 32)

        # Fusion
        x_fused = self.add_layer([x_main, x_branch])

        # Final layers
        x_pool = self.global_avg_pool(x_fused)
        x_flat = self.flatten(x_pool)
        output = self.dense(x_flat)

        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)