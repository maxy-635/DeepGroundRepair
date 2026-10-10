import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First part: 3 pairs of <convolutional layer, max-pooling layer>
        self.conv1 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.pool1 = MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.pool2 = MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv3 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.pool3 = MaxPool2D(pool_size=2, strides=2, padding='valid')

        # Second part: <convolutional layer, Dropout layer, convolutional layer>
        self.conv4 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')
        self.dropout = Dropout(rate=0.5)
        self.conv5 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')

        # Third part: 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
        self.conv6 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
        self.upconv6 = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')
        self.conv7 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.upconv7 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.conv8 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.upconv8 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')

        # Final classification layer
        self.classifier = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First part
        x1 = self.conv1(inputs)
        x1 = self.pool1(x1)
        x2 = self.conv2(x1)
        x2 = self.pool2(x2)
        x3 = self.conv3(x2)
        x3 = self.pool3(x3)

        # Second part
        x = self.conv4(x3)
        x = self.dropout(x)
        x = self.conv5(x)

        # Third part with skip connections
        x = self.conv6(x)
        x = tf.keras.layers.concatenate([x, x3], axis=-1)
        x = self.upconv6(x)
        x = self.conv7(x)
        x = tf.keras.layers.concatenate([x, x2], axis=-1)
        x = self.upconv7(x)
        x = self.conv8(x)
        x = tf.keras.layers.concatenate([x, x1], axis=-1)
        x = self.upconv8(x)

        # Global average pooling and classification
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)