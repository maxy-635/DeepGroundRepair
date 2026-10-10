import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block for each parallel branch
        self.block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

        # Define the parallel convolutional layer
        self.parallel_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # First branch
        branch1 = self.block(x)

        # Second branch (parallel)
        branch2 = self.parallel_conv(x)

        # Combine the outputs using addition
        combined = paddle.add(branch1, branch2)

        # Flatten the combined output
        flattened = paddle.flatten(combined, start_axis=1, stop_axis=-1)

        # Pass through the fully connected layer
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]