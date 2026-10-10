import tensorflow as tf
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Concatenate, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        # Channel group 1: 1x1 separable convolution
        self.sep_conv1x1_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                            padding='same', activation='relu')
        self.sep_conv1x1_2 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                                            padding='same', activation='relu')

        # Channel group 2: 3x3 separable convolution
        self.sep_conv3x3_1 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                            padding='same', activation='relu')
        self.sep_conv3x3_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                                            padding='same', activation='relu')

        # Channel group 3: 5x5 separable convolution
        self.sep_conv5x5_1 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation='relu')
        self.sep_conv5x5_2 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                                            padding='same', activation='relu')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=64, activation='relu')
        self.dense3 = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Split input into three channel groups
        # For CIFAR-10, we'll split the 3 channels into 3 groups of 1 channel each
        # This is a simplified approach - in practice you might want to use Lambda layers
        # to properly split the channels
        group1 = inputs[:, :, :, 0:1]  # First channel
        group2 = inputs[:, :, :, 1:2]  # Second channel
        group3 = inputs[:, :, :, 2:3]  # Third channel

        # Process each group with different kernel sizes
        # Group 1: 1x1 convolutions
        x1 = self.sep_conv1x1_1(group1)
        x1 = self.sep_conv1x1_2(x1)

        # Group 2: 3x3 convolutions
        x2 = self.sep_conv3x3_1(group2)
        x2 = self.sep_conv3x3_2(x2)

        # Group 3: 5x5 convolutions
        x3 = self.sep_conv5x5_1(group3)
        x3 = self.sep_conv5x5_2(x3)

        # Concatenate the outputs
        concatenated = Concatenate(axis=-1)([x1, x2, x3])

        # Global average pooling
        pooled = GlobalAveragePooling2D()(concatenated)

        # Fully connected layers
        x = self.dense1(pooled)
        x = self.dense2(x)
        outputs = self.dense3(x)

        return outputs

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)