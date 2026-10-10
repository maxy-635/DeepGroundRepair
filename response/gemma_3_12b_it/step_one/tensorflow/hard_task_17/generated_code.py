import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, MaxPool2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.dense1 = Dense(units=3072, activation='relu')
        self.dense2 = Dense(units=3072, activation='relu')

        # Block 2
        self.conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # Classifier
        self.dense3 = Dense(units=1024, activation='relu')
        self.dense4 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Block 1
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, shape=(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]))  # Reshape to match input shape

        # Block 2
        y = self.conv1(inputs)
        y = self.conv2(y)
        y = self.maxpool(y)

        # Branch from Block 1 to Block 2
        branch_out = x

        # Fusion
        combined = tf.add(branch_out, y)

        # Classifier
        x = self.dense3(combined)
        output = self.dense4(x)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)