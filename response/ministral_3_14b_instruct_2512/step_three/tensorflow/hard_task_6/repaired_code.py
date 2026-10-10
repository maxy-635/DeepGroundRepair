import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Activation, Lambda, Concatenate, AveragePooling2D, Reshape, Permute

class DL_Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Block 1 components
        self.conv1_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=None)
        self.conv1_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=None)
        self.conv1_3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=None)
        self.activation = Activation('relu')

        # Block 2 components
        self.reshape = Reshape(target_shape=(None, None, 3, 64))
        self.permute = Permute(dims=(1, 2, 4, 3))  # Swap channels and groups
        self.reshape_back = Reshape(target_shape=None)  # Will be set dynamically

        # Block 3
        self.depthwise_conv = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation=None)
        self.activation2 = Activation('relu')

        # Branch path
        self.avg_pool = AveragePooling2D(pool_size=8, strides=8, padding='valid')

        # Final layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Main path
        # Block 1: Split into 3 groups and process each with 1x1 conv
        x1 = self.conv1_1(inputs)
        x2 = self.conv1_2(inputs)
        x3 = self.conv1_3(inputs)

        # Concatenate the three outputs
        x = Concatenate(axis=-1)([x1, x2, x3])

        # Block 2: Channel shuffling
        # Get the shape information
        h, w, c = inputs.shape[1], inputs.shape[2], inputs.shape[3]
        groups = 3
        channels_per_group = c // groups

        # Reshape to (h, w, groups, channels_per_group)
        x = Reshape((h, w, groups, channels_per_group))(x)

        # Permute dimensions to (h, w, channels_per_group, groups)
        x = self.permute(x)

        # Reshape back to original shape
        x = Reshape((h, w, c))(x)

        # Block 3: Depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.activation2(x)

        # Repeat Block 1
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x = Concatenate(axis=-1)([x1, x2, x3])
        x = self.activation(x)

        # Branch path: Average pooling
        branch = self.avg_pool(inputs)

        # Concatenate main path and branch path
        x = Concatenate(axis=-1)([x, branch])

        # Final classification
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Model created successfully!")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)