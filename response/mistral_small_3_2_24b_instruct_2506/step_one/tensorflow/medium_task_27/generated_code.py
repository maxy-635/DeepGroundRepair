import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1 with 3x3 convolution
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        # Branch 2 with 5x5 convolution
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')
        # Global average pooling
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        # Attention mechanism layers
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=2, activation='softmax')
        # Final classification layer
        self.final_dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Branch 1 processing
        branch1 = self.conv1(inputs)
        # Branch 2 processing
        branch2 = self.conv2(inputs)
        # Combine branches by addition
        combined = tf.math.add(branch1, branch2)
        # Global average pooling
        pooled = self.global_avg_pool(combined)
        # Attention mechanism
        attention = self.dense1(pooled)
        attention = self.dense2(attention)
        # Apply attention weights
        branch1_weighted = tf.math.multiply(branch1, attention[:, 0])
        branch2_weighted = tf.math.multiply(branch2, attention[:, 1])
        # Combine weighted branches
        weighted_output = tf.math.add(branch1_weighted, branch2_weighted)
        # Final classification
        output = self.final_dense(weighted_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Model created successfully. Input shape:", input_tensor.shape, "Output shape:", output_tensor.shape)