import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block with three convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Parallel branch with a single convolutional layer
        self.parallel_conv = nn.Conv2D(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Create two blocks
        self.block1 = self._create_block()
        self.block2 = self._create_block()

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128*28*28, out_features=10)  # MNIST images are 28x28

    def _create_block(self):
        """Create a block with three main paths and one parallel path"""
        block = nn.Layer()

        # Main paths for each convolutional layer output
        block.conv1_path = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        block.conv2_path = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        block.conv3_path = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Parallel path
        block.parallel_path = nn.Conv2D(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)

        return block

    def forward(self, x):
        # Process through first block
        block1_out = self.block1(x)
        block1_conv1 = block1_out.conv1_path(x)
        block1_conv2 = block1_out.conv2_path(block1_conv1)
        block1_conv3 = block1_out.conv3_path(block1_conv2)
        block1_parallel = block1_out.parallel_path(x)

        # Combine outputs from first block
        block1_combined = block1_conv1 + block1_conv2 + block1_conv3 + block1_parallel

        # Process through second block
        block2_out = self.block2(x)
        block2_conv1 = block2_out.conv1_path(x)
        block2_conv2 = block2_out.conv2_path(block2_conv1)
        block2_conv3 = block2_out.conv3_path(block2_conv2)
        block2_parallel = block2_out.parallel_path(x)

        # Combine outputs from second block
        block2_combined = block2_conv1 + block2_conv2 + block2_conv3 + block2_parallel

        # Concatenate outputs from both blocks
        combined = paddle.concat([block1_combined, block2_combined], axis=1)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch of 1 MNIST image (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)