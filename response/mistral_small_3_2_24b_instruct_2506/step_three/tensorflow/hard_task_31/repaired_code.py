import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.dropout1 = Dropout(rate=0.2)
        self.conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same')

        # Second block: separable convolutions
        self.sep_conv1 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.dropout2 = Dropout(rate=0.2)
        self.sep_conv2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.dropout3 = Dropout(rate=0.2)
        self.sep_conv3 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')
        self.dropout4 = Dropout(rate=0.2)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block
        main_path = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        main_path = self.dropout1(main_path)  # output tensor shape: (1, 32, 32, 64)
        main_path = self.conv2(main_path)  # output tensor shape of this layer should be (1, 32, 32, 3)
        branch_path = inputs  # output tensor shape: (1, 32, 32, 3)
        output_block1 = tf.math.add(main_path, branch_path)

        # Second block
        group1 = self.sep_conv1(output_block1)
        group1 = self.dropout2(group1)
        group2 = self.sep_conv2(output_block1)
        group2 = self.dropout3(group2)
        group3 = self.sep_conv3(output_block1)
        group3 = self.dropout4(group3)
        output_block2 = tf.experimental.numpy.concatenate([group1, group2, group3], axis=-1)

        # Final layers
        x = self.flatten(output_block2)
        return self.dense(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)