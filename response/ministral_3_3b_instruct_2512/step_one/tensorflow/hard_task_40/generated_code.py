import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, input_tensor):
        # First Block
        x = input_tensor
        x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(x)
        x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
        x = Concatenate(axis=-1)([Flatten()(x1), Flatten()(x2), Flatten()(x3)])
        x = Dense(units=1024, activation='relu')(x)
        x = Reshape(target_shape=(4, 256))(x)

        # Second Block
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')(x)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')(path3)
        path4 = GlobalAveragePooling2D(keepdims=True)(x)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')(path4)
        path1 = Dropout(rate=0.5)(path1)
        path2 = Dropout(rate=0.5)(path2)
        path3 = Dropout(rate=0.5)(path3)
        path4 = Dropout(rate=0.5)(path4)
        x = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Output Layers
        x = Dense(units=512, activation='relu')(x)
        x = Dense(units=10, activation='softmax')(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))
    output_tensor = model(input_tensor)