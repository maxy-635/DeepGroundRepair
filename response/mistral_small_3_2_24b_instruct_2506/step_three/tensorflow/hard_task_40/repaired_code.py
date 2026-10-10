import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout
from tensorflow.keras.models import Sequential

class DL_Model(Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Average pooling layers and concatenation
        self.add(AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid'))

        self.add(Flatten())
        self.add(Flatten())
        self.add(Flatten())

        self.add(Concatenate(axis=-1))
        self.add(Dense(units=1024, activation='relu'))
        self.add(Reshape(target_shape=(16, 16, 4)))

        # Second block: Multi-scale feature extraction
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.add(path1)
        self.add(Dropout(rate=0.5))

        # Path 2: 1x1 + two 3x3 convolutions
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        self.add(path2)
        self.add(Dropout(rate=0.5))

        # Path 3: 1x1 + single 3x3 convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        self.add(path3)
        self.add(Dropout(rate=0.5))

        # Path 4: Average pooling + 1x1 convolution
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        self.add(path4)
        self.add(Dropout(rate=0.5))

        # Concatenate all paths
        self.add(Concatenate(axis=-1)([path1, path2, path3, path4]))

        # Final fully connected layers
        self.add(Flatten())
        self.add(Dense(units=1024, activation='relu'))
        self.add(Dense(units=10, activation='softmax'))

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()