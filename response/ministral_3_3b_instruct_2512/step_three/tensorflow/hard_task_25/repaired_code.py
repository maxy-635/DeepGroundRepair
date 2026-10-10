import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D, Add, Dense, concatenate, Reshape, Lambda

class DL_Model:
    def __init__(self):
        self.main_path = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
            tf.keras.layers.Add()
        ])

        self.branch1 = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        ])

        self.branch2 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')
        ])

        self.branch3 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
            Conv2D(64,  kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation='relu')
        ])

        self.branch_path = tf.keras.Sequential([
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        ])

        # Extract outputs from branches and main path
        self.branch1_output = Lambda(lambda x: x)(self.branch1)
        self.branch2_output = Lambda(lambda x: x)(self.branch2)
        self.branch3_output = Lambda(lambda x: x)(self.branch3)

        self.model = tf.keras.Sequential([
            self.main_path,
            concatenate([self.branch1_output, self.branch2_output, self.branch3_output]),
            Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
            self.branch_path,
            Add(),
            Dense(10, activation='softmax')
        ])

    def __call__(self, input_tensor):
        return self.model(input_tensor)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)