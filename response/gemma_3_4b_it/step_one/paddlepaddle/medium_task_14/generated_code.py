import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential block 1
        self.block1 = nn.Sequential(
            nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )

        # Sequential block 2
        self.block2 = nn.Sequential(
            nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        # Sequential block 3
        self.block3 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust input size based on expected output from convolutional blocks
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Sequential blocks
        block1_output = self.block1(x)
        block2_output = self.block2(block1_output)
        block3_output = self.block3(block2_output)

        # Parallel branch
        parallel_output = self.parallel_branch(x)

        # Concatenate outputs
        aggregated_output = paddle.concat([block1_output, block2_output, block3_output, parallel_output], axis=1)

        # Flatten
        flattened_output = aggregated_output.reshape([-1, 128 * 8 * 8])

        # Fully connected layers
        output = self.fc1(flattened_output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn(1, 3, 32, 32)

    # Pass the input through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output tensor shape:", output_tensor.shape)