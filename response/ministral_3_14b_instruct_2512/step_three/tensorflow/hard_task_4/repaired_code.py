import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Concatenate, Add

class DL_Model(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Initial 1x1 convolution to increase channels threefold
        self.conv1x1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Depthwise separable convolution for feature extraction
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')

        # Channel attention components
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=96, activation='sigmoid')  # Outputs weights for channel attention
        self.reshape = Reshape(target_shape=(-1, 1, 1))  # Reshape to match initial feature dimensions

        # 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Final classification layers
        self.flatten = Flatten()
        self.fc = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Initial processing
        x = self.conv1x1(inputs)
        initial_features = x

        # Feature extraction with depthwise separable convolution
        x = self.depthwise_conv(x)

        # Channel attention mechanism
        channel_attention = self.global_avg_pool(x)
        channel_attention = self.dense1(channel_attention)
        channel_attention = self.dense2(channel_attention)
        channel_attention = self.reshape(channel_attention)

        # Apply channel attention weights
        x = Multiply()([x, channel_attention])

        # Reduce dimensionality and combine with initial input
        x = self.conv1x1_reduce(x)
        x = Add()([x, initial_features])

        # Classification
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)