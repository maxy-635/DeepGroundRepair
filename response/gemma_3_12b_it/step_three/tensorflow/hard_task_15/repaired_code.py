import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Reshape, Add, Activation

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.gap = GlobalAveragePooling2D(keepdims=False)
        self.fc1 = Dense(units=128, activation='relu')
        self.fc2 = Dense(units=32, activation='relu')

        # Branch Path
        self.branch = tf.keras.layers.Input(shape=(32, 32, 3))

        # Combined Path
        self.combined_fc1 = Dense(units=128, activation='relu')
        self.combined_fc2 = Dense(units=10, activation='softmax')  # 10 classes for CIFAR-10

    def call(self, inputs):
        # Main Path
        main_path = self.gap(inputs)
        main_path = self.fc1(main_path)
        main_path = self.fc2(main_path)

        # Reshape main_path to match input shape
        main_path = Reshape((32, 32, 3))(main_path)

        # Branch Path
        branch_path = inputs

        # Add the two paths
        combined_path = Add()([main_path, branch_path])

        # Combined Path
        combined_path = self.combined_fc1(combined_path)
        output = self.combined_fc2(combined_path)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Batch size 1, CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)