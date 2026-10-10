import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential blocks with Conv2D, BatchNorm, and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[32]),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[64]),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[128]),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=[64]),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Process through sequential blocks
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)

        # Process through parallel branch
        parallel_out = self.parallel_branch(x)

        # Add outputs from all paths
        combined = paddle.add(block3_out, parallel_out)

        # Flatten and pass through fully connected layers
        flattened = combined.flatten(start_dim=1)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]