import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')
        self.conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.dropout = Dropout(rate=0.5)

        # Branch pathway
        self.branch_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same')

        # Fusion and final layers
        self.add = Add()
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main pathway
        x = self.conv_3x3(inputs)
        x = self.conv_1x1_1(x)
        x = self.conv_1x1_2(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        # Branch pathway
        branch = self.branch_conv(inputs)

        # Fusion
        x = self.add([x, branch])

        # Final layers
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print(output_tensor)