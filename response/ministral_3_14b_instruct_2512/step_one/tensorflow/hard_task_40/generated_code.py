import tensorflow as tf
from tensorflow.keras.layers import Layer, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block - multi-scale pooling
        self.pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Fully connected layer and reshape
        self.dense = Dense(units=1024, activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 1024))

        # Second block - parallel paths
        # Path 1: 1x1 convolution
        self.path1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Path 2: 1x1 + 3x3 + 3x3 convolutions
        self.path2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.path2_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Path 3: 1x1 + 3x3 convolutions
        self.path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.path3_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

        # Path 4: average pooling + 1x1 convolution
        self.path4_pool = GlobalAveragePooling2D(keepdims=True)
        self.path4_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

        # Dropout layers for each path
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.dropout3 = Dropout(rate=0.5)
        self.dropout4 = Dropout(rate=0.5)

        # Final classification layers
        self.fc1 = Dense(units=256, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # First block processing
        pool1_out = self.pool1(inputs)
        pool2_out = self.pool2(inputs)
        pool3_out = self.pool3(inputs)

        # Flatten and concatenate
        flat1 = Flatten()(pool1_out)
        flat2 = Flatten()(pool2_out)
        flat3 = Flatten()(pool3_out)
        concat = Concatenate(axis=-1)([flat1, flat2, flat3])

        # Dense layer and reshape
        dense_out = self.dense(concat)
        reshaped = self.reshape(dense_out)

        # Second block processing - parallel paths
        # Path 1
        path1 = self.path1_conv1(reshaped)
        path1 = self.dropout1(path1)

        # Path 2
        path2 = self.path2_conv1(reshaped)
        path2 = self.path2_conv2(path2)
        path2 = self.path2_conv3(path2)
        path2 = self.dropout2(path2)

        # Path 3
        path3 = self.path3_conv1(reshaped)
        path3 = self.path3_conv2(path3)
        path3 = self.dropout3(path3)

        # Path 4
        path4 = self.path4_pool(reshaped)
        path4 = self.path4_conv(path4)
        path4 = self.dropout4(path4)

        # Concatenate all paths
        combined = Concatenate(axis=-1)([path1, path2, path3, path4])

        # Final classification
        fc1_out = self.fc1(combined)
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = tf.keras.Input(shape=(28, 28, 1))
    output_tensor = model(input_tensor)
    print("Model created successfully!")
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)