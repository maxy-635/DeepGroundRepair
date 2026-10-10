import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='branch1_conv')
        self.dropout1 = Dropout(rate=0.25, name='branch1_dropout')

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='branch2_conv1')
        self.branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='branch2_conv2')
        self.dropout2 = Dropout(rate=0.25, name='branch2_dropout')

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='branch3_conv1')
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='branch3_conv2')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='branch3_conv3')
        self.dropout3 = Dropout(rate=0.25, name='branch3_dropout')

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = AveragePooling2D(pool_size='mask', strides='mask', padding='mask', name='branch4_pool')
        self.branch4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='branch4_conv')
        self.dropout4 = Dropout(rate=0.25, name='branch4_dropout')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=64, activation='relu', name='dense2')
        self.dense3 = Dense(units=10, activation='softmax', name='dense3')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Process through the four branches
        branch1_out = self.dropout1(self.branch1(inputs))
        print(branch1_out.shape)
        branch2_out = self.dropout2(self.branch2_conv(self.branch2(inputs)))
        branch3_out = self.dropout3(self.branch3_conv2(self.branch3_conv1(self.branch3(inputs))))
        branch4_out = self.dropout4(self.branch4_conv(self.branch4(inputs)))  # output tensor shape of this layer should be (None, 32, 32, 64)

        # Concatenate the outputs of the branches
        concatenated = Concatenate(axis=-1)([branch1_out, branch2_out, branch3_out, branch4_out])

        # Pass through fully connected layers
        dense1_out = self.dense1(concatenated)
        dense2_out = self.dense2(dense1_out)
        output = self.dense3(dense2_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor)