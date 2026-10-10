import paddle
import paddle.nn as nn
import paddle.static as static

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the fully connected layer
        self.fc = static.nn.fc(
            x=None,  # Input will be assigned during the forward pass
            size=10,  # Output size (number of classes in CIFAR-10)
            activation='softmax',
            name='fc'
        )

    def forward(self, x):
        # Get the shape of the input layer
        shape = x.shape

        # Reshape the input tensor into (height, width, groups, channels_per_group)
        groups = 3
        channels_per_group = shape[-1] // groups
        new_shape = (shape[0], shape[1], shape[2], groups, channels_per_group)
        x = paddle.reshape(x, new_shape)

        # Swap the third and fourth dimensions (channel shuffling)
        x = paddle.transpose(x, [0, 1, 3, 2, 4])

        # Reshape back to the original input shape
        x = paddle.reshape(x, shape)

        # Pass the reshaped tensor through the fully connected layer
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()

    # Dummy input tensor (batch_size, height, width, channels)
    batch_size = 2
    height = 32
    width = 32
    channels = 3
    input_tensor = paddle.rand(shape=(batch_size, height, width, channels))

    # Get the output tensor
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)  # Expected: [2, 10]