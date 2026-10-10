import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.concatenate = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups along the channel dimension
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Split the input tensor into three parts
        split_size = channels // 3
        input_split1 = inputs[:, :, :, :split_size]
        input_split2 = inputs[:, :, :, split_size:2*split_size]
        input_split3 = inputs[:, :, :, 2*split_size:]

        # Apply 1x1 convolutions to each group
        conv1 = self.conv1(input_split1)
        conv2 = self.conv2(input_split2)
        conv3 = self.conv3(input_split3)

        # Downsample each group via average pooling
        pool1 = self.avg_pool1(conv1)
        pool2 = self.avg_pool2(conv2)
        pool3 = self.avg_pool3(conv3)

        # Concatenate the three groups along the channel dimension
        concatenated = self.concatenate([pool1, pool2, pool3])

        # Flatten the concatenated feature maps
        flattened = self.flatten(concatenated)

        # Pass through two fully connected layers
        dense1 = self.dense1(flattened)
        output = self.dense2(dense1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape: (32, 32, 3)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)