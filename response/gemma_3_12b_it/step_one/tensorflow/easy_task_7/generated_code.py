import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Add

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout1 = Dropout(rate=0.25)
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.dropout2 = Dropout(rate=0.25)
        self.conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Branch path
        self.branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Flatten and fully connected layer
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        main_path = self.conv1(inputs)
        main_path = self.dropout1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.dropout2(main_path)
        main_path = self.conv3(main_path)

        # Branch path
        branch_path = self.branch_conv(inputs)

        # Combine paths
        combined = Add()([main_path, branch_path])

        # Flatten and dense layer
        flattened = self.flatten(combined)
        output = self.dense(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 28, 28, 1))  # Dummy input tensor with shape (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)