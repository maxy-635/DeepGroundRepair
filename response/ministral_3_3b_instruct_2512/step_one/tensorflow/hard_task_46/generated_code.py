import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, Concatenate, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Split into three groups along the channel, each group uses separable convolutional with different kernel sizes
        self.conv1_1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv1_3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1_5x5 = SeparableConv2D(filters=32, filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

        # Concatenate the outputs from the three groups
        self.concatenate1 = Concatenate(axis=-1)

        # Second block: Multiple branches for enhanced feature extraction
        self.conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_branch1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])
        self.maxpool_branch = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Concatenate the outputs from all branches
        self.concatenate2 = Concatenate(axis=-1)

        # Global average pooling and final dense layer
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x1_1x1 = self.conv1_1x1(inputs)
        x1_3x3 = self.conv1_3x3(inputs)
        x1_5x5 = self.conv1_5x5(inputs)
        x1 = self.concatenate1([x1_1x1, x1_3x3, x1_5x5])

        # Second block
        x2_3x3 = self.conv2_3x3(x1)
        x2_branch1 = self.conv2_branch1(x1)
        x2_maxpool = self.maxpool_branch(x1)
        x2 = self.concatenate2([x2_3x3, x2_branch1, x2_maxpool])

        # Global average pooling and dense layer
        x3 = self.global_avg_pool(x2)
        output = self.dense(x3)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 3, 32, 32))  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)