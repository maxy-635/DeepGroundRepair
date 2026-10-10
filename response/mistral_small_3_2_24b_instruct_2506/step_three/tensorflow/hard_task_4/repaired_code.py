import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution to increase channel dimensionality
        self.conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same')
        # Depthwise separable convolution for feature extraction
        self.depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        # Global average pooling for channel attention
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        # Fully connected layers for channel attention weights
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=3, activation='sigmoid')  # Assuming initial features have 3 channels
        # 1x1 convolution to reduce dimensionality
        self.conv_reduce = Conv2D(filters=1, kernel_size=(1, 1), padding='same')
        # Flatten and dense layers for classification
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # Increase channel dimensionality
        x = self.conv1x1(inputs)
        # Extract initial features
        initial_features = self.depthwise_conv(x)
        # Compute channel attention weights
        pooled = self.global_avg_pool(initial_features)
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        # Reshape and multiply for channel attention
        weights = tf.reshape(weights, (-1, 1, 1, 3))
        attended_features = tf.math.multiply(initial_features, weights)
        # Reduce dimensionality
        reduced = self.conv_reduce(attended_features)
        # Combine with initial input
        combined = tf.math.add(x, reduced)
        # Flatten and classify
        flattened = self.flatten(combined)
        output = self.dense(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)