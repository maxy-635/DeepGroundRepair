import tensorflow as tf
from tensorflow.keras.layers import Conv1D, DepthwiseConv2D, Conv2D, Add, Flatten, Dense

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, input_tensor):
        # Layer 1: 1x1 Conv1D to increase dimensionality
        x = self.model.add(Conv1D(filters=32, kernel_size=1, strides=1, padding='valid', activation='relu')(input_tensor))

        # Layer 2: 3x3 Depthwise Separable Conv2D for feature extraction
        x = self.model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='valid', depth_multiplier=1, activation='relu')(x))

        # Layer 3: 1x1 Conv2D to reduce dimensionality
        x = self.model.add(Conv2D(filters=16, kernel_size=1, strides=1, padding='valid', activation='relu')(x))

        # Layer 4: Add the original input to the processed output
        # Correctly add the original input_tensor to the processed x
        x = self.model.add(Add())([input_tensor, x])

        # Layer 5: Flatten and Dense layer for classification
        x = self.model.add(Flatten())(x)
        x = self.model.add(Dense(units=10, activation='softmax'))(x)

        return x
    
if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 28, 28, 1))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)