import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Flatten, Activation, Multiply, Concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Channel attention path 1: Global Average Pooling + FC layers
        self.gap = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(units=32, activation='relu')
        self.fc2 = Dense(units=64, activation='sigmoid')

        # Channel attention path 2: Global Max Pooling + FC layers
        self.gmp = GlobalMaxPooling2D(keepdims=True)
        self.fc3 = Dense(units=32, activation='relu')
        self.fc4 = Dense(units=64, activation='sigmoid')

        # Spatial attention layers
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.concat = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.fc_final = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # Initial convolution
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Channel attention
        # Path 1: Global Average Pooling
        gap = self.gap(x)  # output tensor shape: (1, 1, 1, 64)
        gap = self.fc1(gap)  # output tensor shape: (1, 1, 1, 32)
        gap = self.fc2(gap)  # output tensor shape: (1, 1, 1, 64)

        # Path 2: Global Max Pooling
        gmp = self.gmp(x)  # output tensor shape: (1, 1, 1, 64)
        gmp = self.fc3(gmp)  # output tensor shape: (1, 1, 1, 32)
        gmp = self.fc4(gmp)  # output tensor shape: (1, 1, 1, 64)

        # Combine paths and apply channel attention
        channel_attention = tf.math.add(gap, gmp)  # output tensor shape: (1, 1, 1, 64)
        channel_attention = Activation('sigmoid')(channel_attention)  # output tensor shape: (1, 1, 1, 64)
        x = tf.math.multiply(x, channel_attention)  # output tensor shape: (1, 32, 32, 64)

        # Spatial attention
        avg = self.avg_pool(x)  # output tensor shape: (1, 64)
        max = self.max_pool(x)  # output tensor shape: (1, 64)
        spatial_attention = self.concat([avg, max])  # output tensor shape: (1, 128)
        spatial_attention = Activation('sigmoid')(spatial_attention)  # output tensor shape: (1, 128)
        x = tf.math.multiply(x, spatial_attention)

        # Final layers
        x = self.flatten(x)
        output = self.fc_final(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Model created successfully. Input shape:", input_tensor.shape, "Output shape:", output_tensor.shape)