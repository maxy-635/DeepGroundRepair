import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, concatenate, Dense

class DL_Model(tf.keras.Model):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 Convolutional Layer
        self.conv1x1_initial = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Branch 1: Local Feature Extraction
        self.conv3x3_1 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.conv3x3_2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Branch 2: Average Pooling and Upsampling
        self.pool2 = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        self.conv3x3_branch2 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.convtranspose3x3_branch2 = Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )

        # Branch 3: Average Pooling and Upsampling
        self.pool2_branch3 = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
        self.conv3x3_branch3 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
        self.convtranspose3x3_branch3 = Conv2DTranspose(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation='relu'
        )
        
        # Concatenation and Refinement
        self.conv1x1_concat = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )

        # Fully Connected Layer
        self.dense = Dense(
            units=10,
            activation='softmax'
        )

    def call(self, inputs):
        # Initial Convolution
        x = self.conv1x1_initial(inputs)

        # Branch 1
        branch1 = self.conv3x3_1(x)
        branch1 = self.conv3x3_2(branch1)

        # Branch 2
        branch2 = self.pool2(x)
        branch2 = self.conv3x3_branch2(branch2)
        branch2 = self.convtranspose3x3_branch2(branch2)

        # Branch 3
        branch3 = self.pool2_branch3(x)
        branch3 = self.conv3x3_branch3(branch3)
        branch3 = self.convtranspose3x3_branch3(branch3)

        # Concatenation
        merged = concatenate([branch1, branch2, branch3], axis=-1)

        # Refinement
        refined = self.conv1x1_concat(merged)

        # Fully Connected Layer
        output = self.dense(refined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = tf.random.normal((1, 32, 32, 3))  # Dummy input tensor (batch_size, height, width, channels)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)