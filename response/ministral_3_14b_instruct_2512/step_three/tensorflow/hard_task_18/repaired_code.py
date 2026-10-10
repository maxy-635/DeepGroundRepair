import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: feature extraction
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        # Second block: channel attention
        self.global_avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 64))
        self.multiply = Multiply()

    def call(self, inputs):
        # First block
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.avg_pool(x)

        # Main path: add input to first block output
        main_path = Add()([inputs, x])

        # Second block: channel attention
        channel_weights = self.global_avg_pool(main_path)
        channel_weights = self.fc1(channel_weights)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.reshape(channel_weights)

        # Multiply input with channel weights
        attention_output = self.multiply([main_path, channel_weights])

        # Flatten and classify
        flattened = Flatten()(attention_output)
        output = Dense(units=10, activation='softmax')(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)