import tensorflow as tf

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups
        self.split_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.split_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        self.split_3 = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')
        
        # Depthwise separable convolutions with different kernel sizes
        self.depthwise_1x1 = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_3x3 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, activation='relu')
        self.depthwise_5x5 = tf.keras.layers.DepthwiseConv2D(kernel_size=5, strides=1, padding='same', depth_multiplier=1, activation='relu')

        # Concatenate the outputs
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        # Flatten and fully connected layer
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input
        split_1 = self.split_1(inputs)
        split_2 = self.split_2(inputs)
        split_3 = self.split_3(inputs)

        # Apply depthwise separable convolutions
        feature_1x1 = self.depthwise_1x1(split_1)
        feature_3x3 = self.depthwise_3x3(split_2)
        feature_5x5 = self.depthwise_5x5(split_3)

        # Concatenate the features
        fused_features = self.concatenate([feature_1x1, feature_3x3, feature_5x5])

        # Flatten and classify
        flattened_features = self.flatten(fused_features)
        output = self.dense(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)