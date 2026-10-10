import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Add, Flatten, Dense

class DL_Model(Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()
        self.num_classes = num_classes

        # Define the three groups of convolutions
        self.conv1x1_group1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')
        self.conv3x3_group1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv1x1_group1_final = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')

        self.conv1x1_group2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')
        self.conv3x3_group2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv1x1_group2_final = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')

        self.conv1x1_group3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')
        self.conv3x3_group3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv1x1_group3_final = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')

        # Define the addition operations
        self.add1 = Add()
        self.add2 = Add()

        # Define the flatten and dense layers
        self.flatten = Flatten()
        self.dense = Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        # Split the input into three groups along the channel dimension
        input_shape = tf.shape(inputs)
        split_size = input_shape[-1] // 3
        group1 = inputs[:, :, :, :split_size]
        group2 = inputs[:, :, :, split_size:2*split_size]
        group3 = inputs[:, :, :, 2*split_size:]

        # Process each group through the convolutional layers
        out_group1 = self.conv1x1_group1(group1)
        out_group1 = self.conv3x3_group1(out_group1)
        out_group1 = self.conv1x1_group1_final(out_group1)

        out_group2 = self.conv1x1_group2(group2)
        out_group2 = self.conv3x3_group2(out_group2)
        out_group2 = self.conv1x1_group2_final(out_group2)

        out_group3 = self.conv1x1_group3(group3)
        out_group3 = self.conv3x3_group3(out_group3)
        out_group3 = self.conv1x1_group3_final(out_group3)

        # Combine the outputs from the three groups
        combined = self.add1([out_group1, out_group2, out_group3])

        # Fuse with the original input
        fused = self.add2([combined, inputs])

        # Flatten and pass through the dense layer
        flattened = self.flatten(fused)
        output = self.dense(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal(shape=(1, 32, 32, 3))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)