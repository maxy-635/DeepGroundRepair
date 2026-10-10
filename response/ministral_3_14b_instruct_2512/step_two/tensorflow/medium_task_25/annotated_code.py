import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Dense, Flatten, BatchNormalization, ReLU, Activation
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Path 1: Single 1x1 convolution
        self.path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2_pool = AveragePooling2D(pool_size='mask', strides='mask', padding='mask')
        self.path2_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 conv followed by parallel 1x3 and 3x1 convs
        self.path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 4: 1x1 conv followed by 3x3 conv, then parallel 1x3 and 3x1 convs
        self.path4_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=128, activation='relu')
        self.classifier = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Path 1
        path1 = self.path1_conv(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Path 2
        path2 = self.path2_pool(inputs)  # output tensor shape of this layer should be (1, 32, 32, 3)
        path2 = self.path2_conv(path2)

        # Path 3
        path3 = self.path3_conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        path3_1x3 = self.path3_conv1x3(path3)  # output tensor shape: (1, 32, 32, 64)
        path3_3x1 = self.path3_conv3x1(path3)  # output tensor shape: (1, 32, 32, 64)
        path3 = Concatenate(axis=-1)([path3_1x3, path3_3x1])  # output tensor shape: (1, 32, 32, 128)

        # Path 4
        path4 = self.path4_conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        path4 = self.path4_conv3x3(path4)  # output tensor shape: (1, 32, 32, 64)
        path4_1x3 = self.path4_conv1x3(path4)  # output tensor shape: (1, 32, 32, 64)
        path4_3x1 = self.path4_conv3x1(path4)  # output tensor shape: (1, 32, 32, 64)
        path4 = Concatenate(axis=-1)([path4_1x3, path4_3x1])  # output tensor shape: (1, 32, 32, 128)

        # Concatenate all paths
        concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Classification
        x = self.flatten(concatenated)
        x = self.dense(x)
        outputs = self.classifier(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)