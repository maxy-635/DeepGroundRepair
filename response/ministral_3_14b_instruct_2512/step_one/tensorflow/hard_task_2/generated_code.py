import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Flatten, Dense, Lambda, Concatenate, Reshape, Activation, BatchNormalization
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First group of convolutions
        self.conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Second group of convolutions
        self.conv1x1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Third group of convolutions
        self.conv1x1_5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final convolution and dense layers
        self.conv_final = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split input into three groups along channel dimension
        input_shape = tf.shape(inputs)
        channels = input_shape[-1]
        split_size = channels // 3

        # Split the input into three parts
        group1 = inputs[:, :, :, :split_size]
        group2 = inputs[:, :, :, split_size:2*split_size]
        group3 = inputs[:, :, :, 2*split_size:]

        # Process each group
        # Group 1
        x1 = self.conv1x1_1(group1)
        x1 = self.conv3x3_1(x1)
        x1 = self.conv1x1_2(x1)

        # Group 2
        x2 = self.conv1x1_3(group2)
        x2 = self.conv3x3_2(x2)
        x2 = self.conv1x1_4(x2)

        # Group 3
        x3 = self.conv1x1_5(group3)
        x3 = self.conv3x3_3(x3)
        x3 = self.conv1x1_6(x3)

        # Combine the three groups
        combined = Add()([x1, x2, x3])

        # Add to original input
        main_path = Add()([combined, inputs])

        # Final convolution
        final_conv = self.conv_final(main_path)

        # Flatten and dense layer
        flattened = self.flatten(final_conv)
        output = self.dense(flattened)

        return output

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