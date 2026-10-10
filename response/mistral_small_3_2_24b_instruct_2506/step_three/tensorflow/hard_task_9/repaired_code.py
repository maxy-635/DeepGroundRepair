import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # 1x1 convolution to adjust output dimensions
        self.adjust_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layers for classification
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

        # Add layer for residual connection
        self.add = Add()

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        branch1_out = self.branch1(inputs)  # output tensor shape: (1, 32, 32, 64)
        branch2_out = self.branch2_conv2(self.branch2_conv1(inputs))  # output tensor shape: (1, 32, 32, 64)
        branch3_out = self.branch3_conv3(self.branch3_conv2(self.branch3_conv1(inputs)))  # output tensor shape: (1, 32, 32, 64)

        # Concatenate the outputs
        concatenated = Concatenate(axis=-1)([branch1_out, branch2_out, branch3_out])  # output tensor shape: (1, 32, 32, 192)

        # Adjust dimensions
        adjusted = self.adjust_conv(concatenated)  # output tensor shape of this layer should be (1, 32, 32, 3)

        # Residual connection
        residual_out = self.add([inputs, adjusted])

        # Flatten and pass through fully connected layers
        flattened = tf.keras.layers.Flatten()(residual_out)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)
        output = self.fc3(fc2_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)