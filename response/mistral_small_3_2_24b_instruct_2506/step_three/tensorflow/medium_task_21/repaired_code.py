import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Concatenate, Dense, Dropout, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = tf.keras.Sequential([
            AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'),
            Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'),
            Dropout(rate=0.2)
        ])

        # Concatenation and fully connected layers
        self.concat = Concatenate(axis=-1)
        self.flatten = Flatten()
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=256, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')  # Assuming CIFAR-10 has 10 classes

    def call(self, inputs):
        # Process each branch
        branch1_out = self.branch1(inputs)
        branch2_out = self.branch2(inputs)
        branch3_out = self.branch3(inputs)
        branch4_out = self.branch4(inputs)

        # Concatenate the outputs
        concat_out = self.concat([branch1_out, branch2_out, branch3_out, branch4_out])

        # Flatten and pass through fully connected layers
        flat_out = self.flatten(concat_out)
        fc1_out = self.fc1(flat_out)
        fc2_out = self.fc2(fc1_out)
        output = self.fc3(fc2_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.keras.Input(shape=(32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print (None, 10)