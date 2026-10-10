import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, GlobalMaxPool2D, Add, Activation, Multiply, AveragePooling2D, MaxPool2D, Concatenate
from tensorflow.keras import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Channel attention path 1: Global Average Pooling
        self.gap1 = GlobalAveragePooling2D(keepdims=True)
        self.dense1_1 = Dense(units=32, activation='relu')
        self.dense1_2 = Dense(units=1, activation='sigmoid')

        # Channel attention path 2: Global Max Pooling
        self.gmp1 = GlobalMaxPool2D(keepdims='mask')
        self.dense2_1 = Dense(units='mask', activation='relu')
        self.dense2_2 = Dense(units=1, activation='sigmoid')

        # Combine channel attention paths
        self.add_channel = Add()
        self.activation_channel = Activation('sigmoid')
        self.multiply_channel = Multiply()

        # Spatial feature extraction
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Concatenate spatial features
        self.concat_spatial = Concatenate(axis=-1)

        # Combine channel and spatial features
        self.multiply_spatial_channel = Multiply()

        # Classification
        self.flatten = tf.keras.layers.Flatten()
        self.dense_class1 = Dense(units=128, activation='relu')
        self.dense_class2 = Dense(units=10, activation='softmax')  # CIFAR-10 has 10 classes

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 64)

        # Channel attention
        gap = self.gap1(x)  # output tensor shape: (1, 1, 1, 64)
        gmp = self.gmp1(x)  # output tensor shape of this layer should be (1, 32, 32, 64)

        gap_out = self.dense1_1(gap)
        gmp_out = self.dense2_1(gmp)  # output tensor shape of this layer should be (1, 1, 1, 64)

        channel_attention = self.activation_channel(self.add_channel([gap_out, gmp_out]))
        x = self.multiply_channel([x, channel_attention])

        # Spatial feature extraction
        avg_spatial = self.avg_pool(x)
        max_spatial = self.max_pool(x)

        spatial_features = self.concat_spatial([avg_spatial, max_spatial])

        # Combine spatial and channel features
        x = self.multiply_spatial_channel([x, spatial_features])

        x = self.flatten(x)
        x = self.dense_class1(x)
        output = self.dense_class2(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)