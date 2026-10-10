import tensorflow as tf
from tensorflow.keras.layers import Conv1D, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Conv2D, Dense, Flatten

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Separable Convolution with different kernel sizes
        self.conv1_1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        self.conv1_3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')
        self.conv1_5x5 = SeparableConv2D(filters=64, filters=64, kernel_size=(5, 5), strides=1, padding='same', activation='relu')
        self.bn1 = BatchNormalization(axis=-1)

        # Block 2: Four parallel branches
        self.path1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        ])

        self.path2 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(3, 3), strides=1, padding='same'),
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')
        ])

        self.path3 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu'),
            tf.keras.Sequential([
                Conv2D(filters=64, kernel_size=(1, 3), strides=1, padding='same', activation='relu'),
                Conv2D(filters=64, kernel_size=(3, 1), strides=1, padding='same', activation='relu')
            ]),
            Concatenate(axis=-1)
        ])

        self.path4 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu'),
            tf.keras.Sequential([
                Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
                tf.keras.Sequential([
                    Conv2D(filters=64, kernel_size=(1, 3), strides=1, padding='same', activation='relu'),
                    Conv2D(filters=64, kernel_size=(3, 1), strides=1, padding='same', activation='relu')
                ]),
                Concatenate(axis=-1)
            ])
        ])

        # Concatenate all paths
        self.concatenate_all = Concatenate(axis=-1)

        # Final layers
        self.flatten = Flatten()
        self.fc = Dense(10, activation='softmax')

    def call(self, inputs):
        # Block 1: Separable Convolution with different kernel sizes
        conv1_1x1 = self.conv1_1x1(inputs)
        conv1_3x3 = self.conv1_3x3(inputs)
        conv1_5x5 = self.conv1_5x5(inputs)
        block1_output = Concatenate(axis=-1)([conv1_1x1, conv1_3x3, conv1_5x5])
        block1_output = self.bn1(block1_output)

        # Block 2: Four parallel branches
        path1_output = self.path1(block1_output)
        path2_output = self.path2(block1_output)
        path3_output = self.path3(block1_output)
        path4_output = self.path4(block1_output)

        # Concatenate all paths
        all_paths_output = self.concatenate_all([path1_output, path2_output, path3_output, path4_output])

        # Final layers
        output = self.flatten(all_paths_output)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)