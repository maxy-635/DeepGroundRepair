import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Dense, Flatten, GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model

class DL_Model:
    def __init__(self):
        # Define the input layer
        self.input_layer = tf.keras.Input(shape=(32, 32, 3))

        # First convolutional layer
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(self.input_layer)

        # Second convolutional layer
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(self.conv1)

        # Third convolutional layer
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(self.conv2)

        # Separate convolutional layer processing input directly
        self.sep_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(self.input_layer)

        # Add outputs of first two conv layers and third conv layer
        self.add_layer = Add()([self.conv1, self.conv2, self.conv3])

        # Flatten the concatenated output
        self.flatten = Flatten()(self.add_layer)

        # Global average pooling
        self.pool = GlobalAveragePooling2D()(self.flatten)

        # First fully connected layer
        self.fc1 = Dense(units=128, activation='relu')(self.pool)

        # Second fully connected layer
        self.fc2 = Dense(units=10, activation='softmax')(self.fc1)

        # Create the model
        self.model = Model(inputs=self.input_layer, outputs=self.fc2)

    def __call__(self, input_tensor):
        return self.model(input_tensor)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)