import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3x3_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')
        self.maxpool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')

        # Second Block
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.dense1 = Dense(units=64, activation='relu')
        self.dense2 = Dense(units=64, activation=None)  # No activation here
        

    def call(self, inputs):
        # First Block
        x = self.conv1x1(inputs)
        y = self.conv3x3_1(inputs)
        z = self.conv5x5(inputs)
        w = self.maxpool(inputs)

        # Concatenate the outputs
        merged = Concatenate(axis=-1)([x, y, z, w])

        # Second Block
        pooled = self.gap(merged)
        dense1_output = self.dense1(pooled)
        weights = self.dense2(dense1_output)

        # Reshape weights to match input shape
        input_shape = tf.shape(inputs)
        reshaped_weights = tf.reshape(weights, [input_shape[0], 1, 1, 64]) # assuming 64 channels after concatenation

        # Multiply weights with input feature map
        weighted_feature_map = tf.math.multiply(merged, reshaped_weights)

        # Final Fully Connected Layer
        final_dense = Dense(units=10, activation='softmax')(weighted_feature_map)

        return final_dense


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)