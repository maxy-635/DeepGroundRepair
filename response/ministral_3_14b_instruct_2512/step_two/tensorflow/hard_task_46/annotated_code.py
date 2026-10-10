import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda, Activation, BatchNormalization

class DL_Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        # First block with three separable convolution branches
        self.sep_conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.sep_conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Second block with multiple branches
        # Branch 1: 3x3 convolution
        self.conv3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 2: 1x1 followed by two 3x3 convolutions
        self.conv1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch 3: Max pooling
        self.max_pool = MaxPool2D(pool_size='mask', strides='mask', padding='mask')

        # Batch normalization layers
        self.bn1 = BatchNormalization(axis=-1)
        self.bn2 = BatchNormalization(axis=-1)

        # Final layers
        self.global_avg_pool = GlobalAveragePooling2D()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block processing
        x1 = self.sep_conv1x1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.sep_conv3x3(inputs)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.sep_conv5x5(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Concatenate outputs from first block
        block1_output = Concatenate(axis=-1)([x1, x2, x3])  # output tensor shape: (1, 32, 32, 192)

        # Second block processing
        # Branch 1
        branch1 = self.conv3x3(block1_output)  # output tensor shape: (1, 32, 32, 128)

        # Branch 2
        branch2 = self.conv1x1(block1_output)  # output tensor shape: (1, 32, 32, 128)
        branch2 = self.conv3x3_1(branch2)  # output tensor shape: (1, 32, 32, 128)
        branch2 = self.conv3x3_2(branch2)  # output tensor shape: (1, 32, 32, 128)

        # Branch 3
        branch3 = self.max_pool(block1_output)  # output tensor shape of this layer should be (1, 32, 32, 192)

        # Concatenate outputs from second block
        block2_output = Concatenate(axis=-1)([branch1, branch2, branch3])

        # Batch normalization
        x = self.bn1(block2_output)
        x = Activation('relu')(x)
        x = self.bn2(x)
        x = Activation('relu')(x)

        # Global average pooling and final classification
        x = self.global_avg_pool(x)
        outputs = self.dense(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)