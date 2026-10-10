import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional block
        self.block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_branch = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Two blocks
        self.block1 = self.block
        self.block2 = self.block

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 7 * 7, out_features=10)  # Assuming input is 28x28 after convolutions


    def forward(self, x):
        # Branch 1
        out_block1 = self.block1(x)

        # Branch 2
        out_block2 = self.block2(x)

        # Parallel branch
        out_parallel = self.parallel_branch(x)

        # Combine the outputs
        combined_output = paddle.add(out_block1, out_parallel)
        combined_output = paddle.add(combined_output, out_block2)

        # Flatten the output
        flattened_output = self.flatten(combined_output)

        # Fully connected layer
        output = self.linear(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)