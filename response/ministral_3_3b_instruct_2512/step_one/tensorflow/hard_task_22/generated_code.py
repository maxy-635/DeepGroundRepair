import tensorflow as tf
from tensorflow.keras.layers import Conv1D, SeparableConv2D, Concatenate, Add, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, MaxPool2D

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Your implementation here

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Example input shape for CIFAR-10 (batch_size=1, height=32, width=32, channels=3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)