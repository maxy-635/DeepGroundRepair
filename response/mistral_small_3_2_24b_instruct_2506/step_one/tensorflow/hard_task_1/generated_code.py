import tensorflow as tf
from tensorflow.keras import layers, Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.initial_conv = layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same')

        # Block 1: Channel attention
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.fc1_path1 = layers.Dense(units=3, activation='relu')
        self.fc2_path1 = layers.Dense(units=3, activation='relu')
        self.fc1_path2 = layers.Dense(units=3, activation='relu')
        self.fc2_path2 = layers.Dense(units=3, activation='relu')
        self.activation = layers.Activation('sigmoid')

        # Block 2: Spatial attention
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.conv1x1 = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same')
        self.sigmoid = layers.Activation('sigmoid')

        # Final classification layer
        self.fc_classifier = layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Initial convolution
        x = self.initial_conv(inputs)

        # Block 1: Channel attention
        # Path 1
        path1 = self.global_avg_pool(x)
        path1 = self.fc1_path1(path1)
        path1 = self.fc2_path1(path1)
        # Path 2
        path2 = self.global_max_pool(x)
        path2 = self.fc1_path2(path2)
        path2 = self.fc2_path2(path2)
        # Combine paths
        combined = tf.math.add(path1, path2)
        attention_weights = self.activation(combined)
        channel_attention = tf.math.multiply(x, attention_weights)

        # Block 2: Spatial attention
        avg = self.avg_pool(channel_attention)
        max = self.max_pool(channel_attention)
        spatial_features = tf.experimental.numpy.concatenate([avg, max], axis=-1)
        spatial_features = self.conv1x1(spatial_features)
        spatial_attention = self.sigmoid(spatial_features)
        spatial_attention = tf.math.multiply(channel_attention, spatial_attention)

        # Additional branch
        branch = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same')(spatial_attention)
        output = tf.math.add(spatial_attention, branch)

        # Final classification
        output = layers.GlobalAveragePooling2D()(output)
        output = self.fc_classifier(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)