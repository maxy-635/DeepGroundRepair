import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, Concatenate, Reshape, Permute, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, Dense

class DL_Model:
    def __init__(self):
        # Block 1: Split into three groups, each processed by 1x1 conv + ReLU
        self.block1_conv1 = Conv1D(filters=1, kernel_size=1, strides=1, padding='same')
        self.block1_conv2 = Conv1D(filters=1, kernel_size=1, strides=1, padding='same')
        self.block1_conv3 = Conv1D(filters=1, kernel_size=1, strides=1, padding='same')
        self.block1_activation = Activation('relu')
        self.block1_concat = Concatenate(axis=-1)

        # Block 2: Channel shuffling
        self.block2_reshape = Reshape(target_shape=(None, None, 3, -1))
        self.block2_permute = Permute(dims=(0, 1, 3, 2))
        self.block2_reshape2 = Reshape(target_shape=(None, None, -1))

        # Block 3: Depthwise separable convolution
        self.block3_conv = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1)

        # Branch path: Global average pooling
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)

        # Final classification
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

    def __call__(self, x):
        # Main path
        # Split into three groups
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x)
        x3 = self.block1_conv3(x)

        # Apply ReLU activation
        x1 = self.block1_activation(x1)
        x2 = self.block1_activation(x2)
        x3 = self.block1_activation(x3)

        # Concatenate
        x_main = self.block1_concat([x1, x2, x3])

        # Block 2: Channel shuffling
        x_main = self.block2_reshape(x_main)
        x_main = self.block2_permute(x_main)
        x_main = self.block2_reshape2(x_main)

        # Block 3: Depthwise separable convolution
        x_main = self.block3_conv(x_main)

        # Branch path: Global average pooling
        x_branch = self.global_avg_pool(x)

        # Concatenate main path and branch path
        x_concat = tf.concat([x_main, x_branch], axis=-1)

        # Final classification
        x_classify = self.flatten(x_concat)
        return self.dense(x_classify)

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # For CIFAR-10, height=32, width=32, channels=3
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)