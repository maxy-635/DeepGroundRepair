import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, Flatten, Dense, Add, concatenate
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.dropout = Dropout(rate=0.5)

        # Branch pathway
        self.branch_conv = Conv2D(filters=64, kernel_size='mask', strides='mask', padding='mask', activation='relu')

        # Final layers
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=False)
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 28, 28, 1)
        # Main pathway
        x = self.conv3x3(inputs)  # output tensor shape: (32, 28, 28, 64)
        x = self.conv1x1_1(x)  # output tensor shape: (32, 28, 28, 32)
        x = self.conv1x1_2(x)  # output tensor shape: (32, 28, 28, 64)
        x = self.max_pool(x)  # output tensor shape: (32, 14, 14, 64)
        x = self.dropout(x)  # output tensor shape: (32, 14, 14, 64)

        # Branch pathway
        branch = self.branch_conv(inputs)  # output tensor shape of this layer should be (32, 14, 14, 64)

        # Add the outputs from both pathways
        x = Add()([x, branch])

        # Final layers
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.random.normal(shape=(32, 28, 28, 1))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)