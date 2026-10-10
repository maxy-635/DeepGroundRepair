import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Concatenate, Add, Flatten, BatchNormalization, Activation, Dropout, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path - three groups with different kernel sizes
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Branch path - 1x1 convolution
        self.branch_conv = Conv2D(filters='mask', kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Batch normalization layers
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn_branch = BatchNormalization()

        # Concatenate and Add layers
        self.concat = Concatenate(axis=-1)
        self.add = Add()

        # Fully connected layers
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

        # Dropout layer
        self.dropout = Dropout(rate=0.5)

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Main path
        x1 = self.conv1x1(inputs)  # output tensor shape: (32, 32, 32, 64)
        x1 = self.bn1(x1)  # output tensor shape: (32, 32, 32, 64)

        x2 = self.conv3x3(inputs)  # output tensor shape: (32, 32, 32, 64)
        x2 = self.bn2(x2)  # output tensor shape: (32, 32, 32, 64)

        x3 = self.conv5x5(inputs)  # output tensor shape: (32, 32, 32, 64)
        x3 = self.bn3(x3)  # output tensor shape: (32, 32, 32, 64)

        # Concatenate the three groups
        concat_output = self.concat([x1, x2, x3])  # output tensor shape: (32, 32, 32, 192)

        # Branch path
        branch_output = self.branch_conv(inputs)  # output tensor shape of this layer should be (32, 32, 32, 192)
        branch_output = self.bn_branch(branch_output)

        # Add main and branch outputs
        fused_features = self.add([concat_output, branch_output])

        # Flatten and pass through fully connected layers
        flattened = Flatten()(fused_features)
        x = self.fc1(flattened)
        x = self.dropout(x)
        outputs = self.fc2(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch size 32, 32x32 images with 3 channels
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)