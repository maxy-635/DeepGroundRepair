import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: Global Average Pooling followed by two fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(32 * 32, 1024)  # Assuming input channels are 3, and after pooling, it's 32*32
        self.fc2 = nn.Linear(1024, 32 * 32)  # Output weight size to match input channels

        # Branch path: 3x3 convolution to match input channels
        self.conv_branch = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Final classification layers
        self.fc3 = nn.Linear(32 * 32, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 10)

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled_flat = paddle.flatten(pooled, 1)  # Flatten the pooled tensor
        weights = self.fc1(pooled_flat)
        weights = self.fc2(weights)
        weights = paddle.reshape(weights, (x.shape[0], 3, 32, 32))  # Reshape to match input shape
        main_path_output = paddle.multiply(x, weights)

        # Branch path
        branch_output = self.conv_branch(x)

        # Combine outputs
        combined_output = paddle.add(main_path_output, branch_output)

        # Final classification layers
        combined_flat = paddle.flatten(combined_output, 1)
        fc3_output = self.fc3(combined_flat)
        fc4_output = self.fc4(fc3_output)
        fc5_output = self.fc5(fc4_output)

        return fc5_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((32, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (32, 10)