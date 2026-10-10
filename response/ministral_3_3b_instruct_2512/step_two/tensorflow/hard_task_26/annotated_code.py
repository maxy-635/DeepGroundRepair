import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Add, Dense, Flatten, concatenate, UpSampling2D, BatchNormalization

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def build(self):
        # Main path
        input_layer = tf.keras.Input(shape=(32, 32, 3))

        # 1x1 initial convolutional
        x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

        # Branch 1: 3x3 convolutional
        branch1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

        # Branch 2: Max pooling + 3x3 convolutional
        branch2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        branch2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: Max pooling + 3x3 convolutional + UpSampling
        branch3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = UpSampling2D(size=(2, 2))(branch3)

        # Concatenate branches
        x = concatenate([branch1, branch2, branch3], axis=-1)

        # Final 1x1 convolutional
        x = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

        # Branch path
        branch_path_input = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        branch_path_input = BatchNormalization(axis=-1)(branch_path_input)

        # Add main path and branch path
        x = Add()([x, branch_path_input])

        # Fully connected layers
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def __call__(self, input_tensor):
        return self.model(input_tensor)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)