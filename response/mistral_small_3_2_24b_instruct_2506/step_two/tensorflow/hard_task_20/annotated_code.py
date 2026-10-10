import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.concatenate = Concatenate(axis=-1)

        # Branch path component
        self.branch_conv = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Addition layer
        self.add = Add()

        # Fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        conv1x1_out = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv3x3_out = self.conv3x3(inputs)  # output tensor shape: (1, 32, 32, 64)
        conv5x5_out = self.conv5x5(inputs)  # output tensor shape: (1, 32, 32, 64)
        concatenated = self.concatenate([conv1x1_out, conv3x3_out, conv5x5_out])  # output tensor shape: (1, 32, 32, 192)

        # Branch path
        branch_out = self.branch_conv(inputs)  # output tensor shape of this layer should be (1, 32, 32, 192)

        # Combine main and branch paths
        fused_features = self.add([concatenated, branch_out])

        # Classification
        x = self.flatten(fused_features)
        x = self.dense1(x)
        output = self.dense2(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)