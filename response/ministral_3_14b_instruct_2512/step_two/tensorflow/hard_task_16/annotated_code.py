import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, GlobalMaxPool2D, Concatenate, Add, Reshape, Lambda, Multiply, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        super(DL_Model, self).__init__()
        self.input_shapes = input_shape
        self.num_classes = num_classes

        # Block 1
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        self.conv1x1_5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Transition Convolution
        self.trans_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Block 2
        self.global_max_pool = GlobalMaxPool2D(keepdims='mask')
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units='mask', activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 64))
        self.multiply = Multiply()

        # Final layers
        self.flatten = Flatten()
        self.fc = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Block 1
        # Group 1
        x1 = self.conv1x1_1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.conv3x3_1(x1)  # output tensor shape: (1, 32, 32, 64)
        x1 = self.conv1x1_2(x1)  # output tensor shape: (1, 32, 32, 64)

        # Group 2
        x2 = self.conv1x1_3(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv3x3_2(x2)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv1x1_4(x2)  # output tensor shape: (1, 32, 32, 64)

        # Group 3
        x3 = self.conv1x1_5(inputs)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.conv3x3_3(x3)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.conv1x1_6(x3)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate
        x = Concatenate(axis=-1)([x1, x2, x3])  # output tensor shape: (1, 32, 32, 192)

        # Transition Convolution
        x = self.trans_conv(x)  # output tensor shape: (1, 32, 32, 64)

        # Branch
        branch = inputs  # output tensor shape: (1, 32, 32, 3)

        # Block 2
        x = self.global_max_pool(x)  # output tensor shape of this layer should be (1, 32, 32, 64)
        x = self.dense1(x)
        x = self.dense2(x)  # output tensor shape of this layer should be (1, 1, 1, 64)
        x = self.reshape(x)
        x = self.multiply([x, x])

        # Add main path and branch
        x = Add()([x, branch])

        # Final layers
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)