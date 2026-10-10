import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Reshape, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Feature extraction
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second block: Channel attention
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation='sigmoid')
        self.reshape = Reshape(target_shape=(1, 1, 64))
        self.flatten = Flatten()

        # Final classification layer
        self.classifier = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.avg_pool(x)

        # Second block
        channel_weights = self.global_avg_pool(x)
        channel_weights = self.dense1(channel_weights)
        channel_weights = self.dense2(channel_weights)
        channel_weights = self.reshape(channel_weights)

        # Multiply input with channel weights
        x = tf.math.multiply(x, channel_weights)

        # Flatten and classify
        x = self.flatten(x)
        output = self.classifier(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model output shape:", output_tensor.shape)