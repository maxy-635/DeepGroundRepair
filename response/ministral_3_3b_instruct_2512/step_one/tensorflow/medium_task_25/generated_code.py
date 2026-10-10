import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool2D, Conv2DTranspose, GlobalAveragePooling2D, Flatten, Dense, Concatenate, Reshape, BatchNormalization

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def build_model(self):
        # Path 1: single 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(self.input_tensor)

        # Path 2: average pooling followed by 1x1 convolution
        path2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(self.input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(path2)

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(self.input_tensor)
        path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate(axis=-1)([path3_1x3, path3_3x1])

        # Path 4: 1x1 convolution followed by 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(self.input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate(axis=-1)([path4_1x3, path4_3x1])

        # Concatenate all paths
        output = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Global average pooling and dense layer for classification
        output = GlobalAveragePooling2D()(output)
        output = Dense(units=10, activation='softmax')(output)

        return output

    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        return self.build_model()

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)