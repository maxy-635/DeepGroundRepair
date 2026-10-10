import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: Average pooling followed by 1x1 convolution
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions and concatenates the results
        self.conv3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions and concatenates the results
        self.conv4_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv4_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')

        # Fully connected layer for classification
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Path 1
        path1 = self.conv1_1x1(inputs)

        # Path 2
        path2 = self.avg_pool(inputs)
        path2 = self.conv2_1x1(path2)

        # Path 3
        path3 = self.conv3_1x1(inputs)
        path3_1x3 = self.conv3_1x3(path3)
        path3_3x1 = self.conv3_3x1(path3)
        path3 = Concatenate(axis=-1)([path3_1x3, path3_3x1])

        # Path 4
        path4 = self.conv4_1x1(inputs)
        path4 = self.conv4_3x3(path4)
        path4_1x3 = self.conv4_1x3(path4)
        path4_3x1 = self.conv4_3x1(path4)
        path4 = Concatenate(axis=-1)([path4_1x3, path4_3x1])

        # Feature fusion
        fused_features = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Flatten the features
        fused_features = tf.keras.layers.Flatten()(fused_features)

        # Classification
        output = self.dense(fused_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)