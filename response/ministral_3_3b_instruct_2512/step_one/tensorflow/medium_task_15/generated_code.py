import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, Conv2DTranspose, AveragePooling2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.bn1 = BatchNormalization(axis=-1)
        self.relu1 = ReLU()

        # Global average pooling and fully connected layers
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.dense1 = Dense(units=32, activation=None)
        self.dense2 = Dense(units=32, activation=None)

        # Reshape and multiply layers
        self.reshape = Reshape(target_shape=(32, 32, 32))
        self.multiply = Multiply()
        self.concatenate = Concatenate(axis=-1)

        # Downsampling layers
        self.conv2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')

        # Final fully connected layer
        self.final_dense = Dense(units=10, activation=None)

    def call(self, inputs):
        # Initial feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global average pooling and fully connected layers
        pooled = self.gap(x)
        pooled = self.dense1(pooled)
        pooled = self.dense2(pooled)

        # Reshape and multiply
        reshaped = self.reshape(pooled)
        weighted = self.multiply([x, reshaped])

        # Concatenate with original input
        concatenated = self.concatenate([inputs, weighted])

        # Downsampling
        x = self.conv2(concatenated)
        x = self.pool2(x)

        # Final classification
        output = self.final_dense(x)

        return output

if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)