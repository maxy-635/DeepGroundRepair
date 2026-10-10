import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Define the parallel branch
        self.parallel_conv = nn.Conv2D(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.parallel_relu = nn.ReLU()

        # Define the two parallel branches
        self.branch1 = self.conv_block
        self.branch2 = self.conv_block

        # Define the flattening and fully connected layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass input through the first branch
        branch1_output = self.branch1(x)  # output tensor shape: (1, 128, 28, 28)

        # Pass input through the parallel branch
        parallel_output = self.parallel_relu(self.parallel_conv(x))  # output tensor shape: (1, 128, 28, 28)

        # Add the outputs of the two branches
        combined_output = paddle.add(branch1_output, parallel_output)  # output tensor shape: (1, 128, 28, 28)

        # Pass input through the second branch
        branch2_output = self.branch2(x)  # output tensor shape: (1, 128, 28, 28)

        # Concatenate the outputs of the two blocks
        concatenated_output = paddle.concat([combined_output, branch2_output], axis=1)  # output tensor shape: (1, 256, 28, 28)

        # Flatten the concatenated output
        flattened_output = self.flatten(concatenated_output)  # output tensor shape: (1, 200704)

        # Pass the flattened output through the fully connected layer
        linear_output = self.linear(flattened_output)

        # Apply softmax to get the classification probabilities
        output = self.softmax(linear_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)