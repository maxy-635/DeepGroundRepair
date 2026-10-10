import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Conv2D, Add, Flatten

class DL_Model:
    def __init__(self):
        self.main_path = tf.keras.Sequential([
            GlobalAveragePooling2D(keepdims=True),
            Dense(units=1024, activation='relu'),
            Dense(units=1024, activation='relu'),
            Reshape(target_shape=(32, 32, 1024))
        ])

        self.branch_path = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.combine = Add()

    def __call__(self, input_tensor):
        main_output = self.main_path(input_tensor)
        branch_output = self.branch_path(input_tensor)
        combined_output = self.combine([main_output, branch_output])
        return combined_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)