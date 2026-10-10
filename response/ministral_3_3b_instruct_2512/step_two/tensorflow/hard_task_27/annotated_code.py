import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, LayerNormalization, Dense, Add, Flatten, GlobalAveragePooling2D, Conv2D, BatchNormalization

class DL_Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def __call__(self, input_tensor):
        x = self.model(input_tensor)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)