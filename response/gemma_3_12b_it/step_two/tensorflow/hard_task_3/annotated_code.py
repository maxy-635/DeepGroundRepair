import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Dropout, Concatenate, Add, Dense, BatchNormalization, Input
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv2d_1_main_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=relu)
        self.conv2d_3_1_main_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
        self.dropout_1_main = Dropout(rate=0.25)
        self.conv2d_1_main_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=relu)
        self.conv2d_3_1_main_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
        self.dropout_2_main = Dropout(rate=0.25)

        # Branch Pathway
        self.conv2d_1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=relu)

        # Input Layer
        self.input_layer = Input(shape=(32, 32, 3), dtype='float32', name='input')

    def call(self, inputs):
        # Main Pathway
        x_main = self.conv2d_1_main_1(inputs)
        x_main = self.conv2d_3_1_main_1(x_main)
        x_main = self.dropout_1_main(x_main)
        x_main = self.conv2d_1_main_2(x_main)
        x_main = self.conv2d_3_1_main_2(x_main)
        x_main = self.dropout_2_main(x_main)

        # Branch Pathway
        x_branch = self.conv2d_1_branch(inputs)

        # Combine Pathways
        combined = Add()([x_main, x_branch])

        # Classification
        x = Dense(units=10, activation='softmax')(combined)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)