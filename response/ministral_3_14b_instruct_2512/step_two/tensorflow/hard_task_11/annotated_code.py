import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Concatenate, Add, Lambda, Flatten, Activation, BatchNormalization, ReLU, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Main pathway components
        self.conv1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x3_branch = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x1_branch = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Direct connection components
        self.shortcut_conv = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Classification components
        self.pool = GlobalAveragePooling2D()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main pathway
        x = self.conv1x1_main(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Parallel branch
        branch = self.conv1x1_branch(inputs)  # output tensor shape: (1, 32, 32, 64)
        branch = Concatenate(axis=-1)([branch, self.conv1x3_branch(branch), self.conv3x1_branch(branch)])  # output tensor shape: (1, 32, 32, 192)
        branch = self.conv1x1_output(branch)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate main pathway and parallel branch
        x = Concatenate(axis=-1)([x, branch])  # output tensor shape: (1, 32, 32, 128)

        # Direct connection (shortcut)
        shortcut = self.shortcut_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 128)
        x = Add()([x, shortcut])

        # Classification
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)