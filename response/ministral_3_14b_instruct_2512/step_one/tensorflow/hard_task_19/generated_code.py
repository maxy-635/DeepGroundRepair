import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Branch path
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=256, activation='relu')

        # Final layers
        self.flatten = Flatten()
        self.fc1 = Dense(units=512, activation='relu')
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)

        # Branch path
        branch = self.global_avg_pool(x)
        branch = self.dense1(branch)
        branch = self.dense2(branch)

        # Reshape and multiply
        channel_weights = Reshape((1, 1, 256))(branch)
        weighted_x = Multiply()([x, channel_weights])

        # Combine paths
        combined = Add()([x, weighted_x])
        combined = self.flatten(combined)

        # Final classification layers
        output = self.fc1(combined)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.keras.Input(shape=(32, 32, 3))
    output_tensor = model(input_tensor)
    print("Model summary:")
    print(model.summary())