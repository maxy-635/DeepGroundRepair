import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Add, Concatenate, Dense, Lambda, Reshape, Activation, BatchNormalization, GlobalAveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: dual-path structure
        self.conv1_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2_main = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Second block: three groups with depthwise separable convolutions
        self.depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

        # Batch normalization layers
        self.batch_norm1 = BatchNormalization(axis=-1)
        self.batch_norm2 = BatchNormalization(axis=-1)

    def call(self, inputs):
        # First block: dual-path structure
        main_path = self.conv1_main(inputs)
        main_path = self.conv2_main(main_path)
        main_path = self.conv3_main(main_path)
        branch_path = inputs
        output_block1 = Add()([main_path, branch_path])

        # Second block: three groups with depthwise separable convolutions
        # Split input into three groups along channel dimension
        input_shape = tf.shape(inputs)
        channels = input_shape[-1]
        group_size = channels // 3

        # Lambda function to split the input into three groups
        def split_channels(x):
            x1 = x[:, :, :, :group_size]
            x2 = x[:, :, :, group_size:2*group_size]
            x3 = x[:, :, :, 2*group_size:]
            return [x1, x2, x3]

        x1, x2, x3 = tf.split(inputs, num_or_size_splits=3, axis=-1)

        # Apply depthwise separable convolutions
        x1 = self.depthwise_conv1(x1)
        x2 = self.depthwise_conv2(x2)
        x3 = self.depthwise_conv3(x3)

        # Concatenate the outputs
        output_block2 = Concatenate(axis=-1)([x1, x2, x3])

        # Combine the outputs from both blocks
        combined = Add()([output_block1, output_block2])

        # Global average pooling and flatten
        pooled = GlobalAveragePooling2D()(combined)

        # Fully connected layers
        fc1 = self.dense1(pooled)
        fc2 = self.dense2(fc1)

        return fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)