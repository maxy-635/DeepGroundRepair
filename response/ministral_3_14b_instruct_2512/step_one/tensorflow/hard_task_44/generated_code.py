import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Concatenate, MaxPool2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel convolutional branches with different kernel sizes
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.2)
        self.concat1 = Concatenate(axis=-1)

        # Block 2: Four parallel branches with different operations
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.conv1x1_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.concat2 = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1 processing
        x1 = self.conv1x1_1(inputs)
        x2 = self.conv3x3_1(inputs)
        x3 = self.conv5x5_1(inputs)

        # Concatenate outputs from Block 1
        x = self.concat1([x1, x2, x3])
        x = self.dropout1(x)

        # Block 2 processing - four branches
        # Branch 1: 1x1 conv
        branch1 = self.conv1x1_2(x)

        # Branch 2: 1x1 conv + 3x3 conv
        branch2 = self.conv1x1_3(x)
        branch2 = self.conv3x3_2(branch2)

        # Branch 3: 1x1 conv + 5x5 conv
        branch3 = self.conv1x1_3(x)
        branch3 = self.conv5x5_2(branch3)

        # Branch 4: 3x3 max pooling + 1x1 conv
        branch4 = self.maxpool(x)
        branch4 = self.conv1x1_4(branch4)

        # Concatenate all branches
        x = self.concat2([branch1, branch2, branch3, branch4])

        # Final layers
        x = self.flatten(x)
        outputs = self.dense(x)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print("Model summary:")
    model.summary()