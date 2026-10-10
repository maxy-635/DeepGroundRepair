import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.bn1x1 = BatchNormalization(axis=-1)
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.bn3x3 = BatchNormalization(axis=-1)
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.bn5x5 = BatchNormalization(axis=-1)
        self.concat1 = Concatenate(axis=-1)

        # Block 2
        # Path 1
        self.path1_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        # Path 2
        self.path2_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.path2_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        # Path 3
        self.path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)
        # Path 4
        self.path4_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.path4_conv3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat3 = Concatenate(axis=-1)
        self.concat4 = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x1 = self.conv1x1(inputs)
        x1 = self.bn1x1(x1)
        x2 = self.conv3x3(inputs)
        x2 = self.bn3x3(x2)
        x3 = self.conv5x5(inputs)
        x3 = self.bn5x5(x3)
        x = self.concat1([x1, x2, x3])

        # Block 2
        # Path 1
        p1 = self.path1_conv(x)
        # Path 2
        p2 = self.path2_pool(x)
        p2 = self.path2_conv(p2)
        # Path 3
        p3 = self.path3_conv1(x)
        p3_1 = self.path3_conv1x3(p3)
        p3_2 = self.path3_conv3x1(p3)
        p3 = self.concat2([p3_1, p3_2])
        # Path 4
        p4 = self.path4_conv1(x)
        p4 = self.path4_conv3x3(p4)
        p4_1 = self.path4_conv1x3(p4)
        p4_2 = self.path4_conv3x1(p4)
        p4 = self.concat3([p4_1, p4_2])
        # Concatenate all paths
        x = self.concat4([p1, p2, p3, p4])

        # Final layers
        x = self.flatten(x)
        output = self.dense(x)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)