import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Add, Reshape

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.gap = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.fc3 = Dense(units=10, activation='softmax')

        # Branch path
        self.reshape = Reshape(target_shape=(128, 1))

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        # Branch path
        branch = self.gap(inputs)
        branch = self.fc1(branch)
        branch = self.fc2(branch)
        branch = self.reshape(branch)

        # Weighted sum
        weighted_sum = tf.multiply(x, branch)
        x = tf.add(x, weighted_sum)

        # Final classification
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)