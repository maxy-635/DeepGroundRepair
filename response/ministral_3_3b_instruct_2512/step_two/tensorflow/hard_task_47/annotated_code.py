import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def build(self):
        # First block: Split into three groups with different kernel sizes
        x = tf.keras.Input(shape=(32, 32, 3))
        x1 = SeparableConv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        x1 = BatchNormalization(axis=-1)(x1)
        x2 = SeparableConv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x2 = BatchNormalization(axis=-1)(x2)
        x3 = SeparableConv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='valid')(x)
        x3 = BatchNormalization(axis=-1)(x3)

        # Concatenate the outputs
        x = Concatenate()([x1, x2, x3])

        # Second block: Multiple branches for feature extraction
        branch1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        branch1 = BatchNormalization(axis=-1)(branch1)

        branch2 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        branch2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
        branch2 = BatchNormalization(axis=-1)(branch2)

        branch3 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        branch3 = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same')(branch3)
        branch3 = Conv2D(64, kernel_size=(1, 7), strides=(1, 1), padding='same')(branch3)
        branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
        branch3 = BatchNormalization(axis=-1)(branch3)

        branch4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

        # Concatenate the outputs from all branches
        x = Concatenate()([branch1, branch2, branch3, branch4])

        # Flatten and classify
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=x, outputs=x)

    def __call__(self, input_tensor):
        return self.model(input_tensor)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)