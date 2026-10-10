import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Concatenate, Lambda, DepthwiseConv2D, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Split into three groups, apply 1x1 conv, and concatenate
        self.conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat1 = Concatenate(axis=-1)

        # Block 2: Channel shuffling
        self.reshape2 = Lambda(lambda x: tf.reshape(x, (-1, x.shape[1], x.shape[2], 3, x.shape[-1] // 3)))
        self.permute2 = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))
        self.reshape_back2 = Lambda(lambda x: tf.reshape(x, (-1, x.shape[1], x.shape[2], x.shape[-1])))

        # Block 3: Depthwise separable convolution
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Branch path: Average pooling
        self.avg_pool = AveragePooling2D(pool_size=(32, 32), strides=(1, 1), padding='valid')

        # Final concatenation and dense layer
        self.concat_final = Concatenate(axis=-1)
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        # Split into three groups (assuming input has 96 channels after Block 1)
        x1 = self.conv1_1(inputs)
        x2 = self.conv1_2(inputs)
        x3 = self.conv1_3(inputs)
        x = self.concat1([x1, x2, x3])

        # Block 2: Channel shuffling
        x = self.reshape2(x)
        x = self.permute2(x)
        x = self.reshape_back2(x)

        # Block 3: Depthwise separable convolution
        x = self.depthwise_conv3(x)

        # Block 1 repeated
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x = self.concat1([x1, x2, x3])

        # Branch path
        branch = self.avg_pool(inputs)

        # Concatenate main path and branch path
        combined = self.concat_final([x, branch])

        # Final classification
        output = self.dense(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)