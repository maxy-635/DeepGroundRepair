import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')
        self.pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten(data_format='channels_last')
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (32, 32, 32, 3)
        # Split the input into three groups along the channel dimension
        group1, group2, group3 = tf.split(inputs, num_or_size_splits=3, axis=-1)  # output tensor shape: (32, 32, 32, 1); (32, 32, 32, 1); (32, 32, 32, 1)

        # Apply 1x1 convolutions to each group
        group1 = self.conv1(group1)  # output tensor shape: (32, 32, 32, 32)
        group2 = self.conv2(group2)  # output tensor shape: (32, 32, 32, 32)
        group3 = self.conv3(group3)  # output tensor shape: (32, 32, 32, 32)

        # Downsample each group via average pooling
        group1 = self.pool(group1)  # output tensor shape: (32, 16, 16, 32)
        group2 = self.pool(group2)  # output tensor shape: (32, 16, 16, 32)
        group3 = self.pool(group3)  # output tensor shape: (32, 16, 16, 32)

        # Concatenate the three groups along the channel dimension
        concatenated = self.concatenate([group1, group2, group3])  # output tensor shape: (32, 16, 16, 96)

        # Flatten and pass through fully connected layers
        flattened = self.flatten(concatenate)
        dense1 = self.dense1(flattened)
        outputs = self.dense2(dense1)

        return outputs

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(32, 32, 32, 3))  # Batch size 32, height 32, width 32, channels 32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)