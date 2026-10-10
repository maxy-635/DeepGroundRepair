import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Add, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_main')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='conv3x3_main')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', name='conv5x5_main')

        # Branch path
        self.conv1x1_branch = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1x1_branch')

        # Classification layers
        self.dense1 = Dense(units=128, activation='relu', name='dense1')
        self.dense2 = Dense(units=10, activation='softmax', name='dense2')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Main path
        x1 = self.conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv3x3(inputs)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.conv5x5(inputs)  # output tensor shape: (1, 32, 32, 64)
        merged = Concatenate(axis=-1)([x1, x2, x3])  # output tensor shape: (1, 32, 32, 192)

        # Branch path
        x_branch = self.conv1x1_branch(inputs)  # output tensor shape of this layer should be (1, 32, 32, 192)

        # Fuse features
        fused_features = Add()([merged, x_branch])

        # Classification
        x = self.dense1(fused_features)
        output = self.dense2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, height, width, channels)
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example: batch size 1, 32x32 image, 3 channels (RGB)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)