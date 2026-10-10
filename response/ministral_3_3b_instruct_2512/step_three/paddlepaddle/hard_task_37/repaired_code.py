import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Define the parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Define the block
        self.block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Define the concatenation and final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Process the input through the block
        block_output = self.block(x)  # output tensor shape of this layer should be (1, 64, 28, 28)

        # Process the input through the parallel branch
        parallel_output = self.parallel_branch(x)  # output tensor shape: (1, 64, 28, 28)

        # Add the outputs from the block and parallel branch
        added_output = paddle.add(block_output, parallel_output)

        # Concatenate the outputs from the two blocks
        concatenated_output = paddle.concat([block_output, added_output], axis=1)

        # Flatten and pass through the fully connected layer
        flattened_output = self.flatten(concatenated_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)