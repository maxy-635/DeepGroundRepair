import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Concatenate, Add, Flatten, Reshape, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: 1x1 conv followed by two 3x3 convs
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Concatenate and 1x1 conv
        self.concat = Concatenate(axis=-1)
        self.conv_after_concat = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Main path (identity connection)
        self.identity = Add()

        # Classification head
        self.flatten = Flatten()
        self.dense1 = Dense(units=512, activation='relu')
        self.dropout1 = Dropout(rate=0.5)
        self.dense2 = Dense(units=256, activation='relu')
        self.dropout2 = Dropout(rate=0.5)
        self.dense3 = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Main path
        x = inputs

        # Branch 1
        branch1 = self.branch1(x)

        # Branch 2
        branch2 = self.branch2_conv2(self.branch2_conv1(x))

        # Branch 3
        branch3 = self.branch3_conv3(self.branch3_conv2(self.branch3_conv1(x)))

        # Concatenate branches
        concat = self.concat([branch1, branch2, branch3])

        # 1x1 conv after concatenation
        concat = self.conv_after_concat(concat)

        # Add to identity path
        x = self.identity([x, concat])

        # Classification head
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)