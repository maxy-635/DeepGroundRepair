import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Concatenate, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Split into three groups with different kernel sizes
        self.conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.concat1 = Concatenate(axis=-1)

        # Second block: Multiple branches for enhanced feature extraction
        self.branch1_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch2_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.branch3_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.concat2 = Concatenate(axis=-1)

        # Final layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x1 = self.conv1x1(inputs)
        x2 = self.conv3x3(inputs)
        x3 = self.conv5x5(inputs)
        x = self.concat1([x1, x2, x3])

        # Second block
        branch1 = self.branch1_conv(x)
        branch2 = self.branch2_conv3(self.branch2_conv2(self.branch2_conv1(x)))
        branch3 = self.branch3_pool(x)
        x = self.concat2([branch1, branch2, branch3])

        # Final layers
        x = self.global_avg_pool(x)
        outputs = self.dense(x)
        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor)