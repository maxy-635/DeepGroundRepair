import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Dense, Flatten, concatenate, BatchNormalization, Activation, MaxPool2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First path: three sequential convolutional layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Second path: direct convolutional layer
        self.conv_direct = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Pooling layer
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Batch normalization layers
        self.bn1 = BatchNormalization(axis=-1)
        self.bn2 = BatchNormalization(axis=-1)
        self.bn3 = BatchNormalization(axis=-1)
        self.bn_direct = BatchNormalization(axis=-1)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First path processing
        x1 = self.conv1(inputs)
        x1 = self.bn1(x1)
        x1 = self.max_pool(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.max_pool(x2)

        x3 = self.conv3(x2)

        # Second path processing
        x_direct = self.conv_direct(inputs)
        x_direct = self.bn_direct(x_direct)

        # Add outputs from first two conv layers with the third conv layer
        add_output = Add()([x2, x3])

        # Concatenate the add_output with direct path output
        concat_output = concatenate([add_output, x_direct], axis=-1)

        # Flatten and pass through fully connected layers
        x = self.flatten(concat_output)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape: 32x32x3
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)