import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

class DL_Model(Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the pooling layers with different window sizes and strides
        self.pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')

        # Flatten layers for each pooling output
        self.flatten1 = Flatten()
        self.flatten2 = Flatten()
        self.flatten3 = Flatten()

        # Concatenate layer
        self.concatenate = Concatenate(axis=-1)

        # Fully connected layers
        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Apply pooling layers
        pool1_out = self.pool1(inputs)
        pool2_out = self.pool2(inputs)
        pool3_out = self.pool3(inputs)

        # Flatten each output
        flat1 = self.flatten1(pool1_out)
        flat2 = self.flatten2(pool2_out)
        flat3 = self.flatten3(pool3_out)

        # Concatenate the flattened features
        concatenated = self.concatenate([flat1, flat2, flat3])

        # Process through fully connected layers
        x = self.dense1(concatenated)
        outputs = self.dense2(x)

        return outputs

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create a dummy input tensor with shape (batch_size, height, width, channels)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))

    # Get model output
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)