import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Add, SeparableConv2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch_conv = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Second block: separable convolutions with different kernel sizes
        self.sep_conv1 = SeparableConv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv2 = SeparableConv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv3 = SeparableConv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, inputs):
        # First block: main path and branch path
        main_path = self.dropout1(self.conv1(inputs))
        main_path = self.conv2(main_path)
        branch_path = self.branch_conv(inputs)
        output = Add()([main_path, branch_path])

        # Second block: separable convolutions with different kernel sizes
        sep_conv1 = self.sep_conv1(output)
        sep_conv2 = self.sep_conv2(output)
        sep_conv3 = self.sep_conv3(output)

        # Concatenate the outputs from the three separable convolutions
        concatenated = tf.concat([sep_conv1, sep_conv2, sep_conv3], axis=-1)

        # Flatten and dense layers
        output = self.flatten(concatenated)
        output = self.dense(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)