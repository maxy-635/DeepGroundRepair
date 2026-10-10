import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Flatten
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters='mask', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')
        self.avg_pool = AveragePooling2D(pool_size='mask', strides='mask', padding='mask')

        # Second block
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.fc1 = Dense(units=64, activation='relu')
        self.fc2 = Dense(units=64, activation='relu')
        self.reshape = Reshape(target_shape=(1, 1, 64))

        self.flatten = Flatten()
        self.fc3 = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # the initial shape of inputs is: (1, 32, 32, 3)
        # First block
        x = self.conv1(inputs)  # output tensor shape: (1, 32, 32, 64)
        x = self.conv2(x)  # output tensor shape of this layer should be (1, 32, 32, 3)
        x = self.avg_pool(x)  # output tensor shape of this layer should be (1, 32, 32, 64)

        # Main path
        main_path = inputs  # output tensor shape: (1, 32, 32, 3)
        
        # Combine the first block and main path
        combined = Add()([x, main_path])

        # Second block
        channel_weights = self.gap(combined)
        channel_weights = self.fc1(channel_weights)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.reshape(channel_weights)

        weighted_features = combined * channel_weights
        
        # Classification
        flattened = self.flatten(weighted_features)
        output = self.fc3(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 input shape: (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model compiled successfully.")