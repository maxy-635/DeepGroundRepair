import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Dense, Flatten, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 1
        self.branch1_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2
        self.branch2_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_upsample = UpSampling2D(size=(2, 2))

        # Branch 3
        self.branch3_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_upsample = UpSampling2D(size=(2, 2))

        # Concatenate and final 1x1 conv
        self.concat = Concatenate(axis=-1)
        self.main_conv1x1_final = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.branch_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.main_conv1x1(inputs)

        # Branch 1
        branch1 = self.branch1_conv3x3(x)

        # Branch 2
        branch2 = self.branch2_maxpool(x)
        branch2 = self.branch2_conv3x3(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_maxpool(x)
        branch3 = self.branch3_conv3x3(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate and final 1x1 conv
        concat = self.concat([branch1, branch2, branch3])
        main_output = self.main_conv1x1_final(concat)

        # Branch path
        branch_output = self.branch_conv1x1(inputs)

        # Add main and branch outputs
        add_output = Add()([main_output, branch_output])

        # Fully connected layers
        x = self.flatten(add_output)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (None, 10)