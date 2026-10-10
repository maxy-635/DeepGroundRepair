import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dropout, Flatten, Dense

class DL_Model(Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path layers
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
        self.dropout1 = Dropout(rate=0.25)
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.dropout2 = Dropout(rate=0.25)
        self.conv3 = Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')

        # Flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(units=10, activation='softmax')

    def call(self, inputs):
        # Main path
        x = self.conv1(inputs)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)

        # Branch path (identity connection)
        branch = inputs

        # Combine main path and branch path
        combined = tf.math.add(x, branch)

        # Final classification layers
        x = self.flatten(combined)
        x = self.dense(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Dummy input tensor for MNIST
    output_tensor = model(input_tensor)
    print("Model created successfully. Input shape:", input_tensor.shape, "Output shape:", output_tensor.shape)