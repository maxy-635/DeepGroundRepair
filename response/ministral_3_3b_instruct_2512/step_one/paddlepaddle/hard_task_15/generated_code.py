import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: Global Average Pooling followed by two fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1_main = nn.Linear(32 * 32 * 10, 1024)  # Assuming input is 32x32x10 channels
        self.fc2_main = nn.Linear(1024, 32 * 32 * 10)

        # Branch path: Direct connection to input without modification
        self.branch_path = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 10, 1024),
            nn.Linear(1024, 32 * 32 * 10)
        )

        # Final output layer
        self.final_fc1 = nn.Linear(32 * 32 * 20, 10)  # Combined output from main and branch paths

    def forward(self, x):
        # Main path
        x_main = self.global_avg_pool(x)
        x_main = x_main.flatten(start_dim=1)
        x_main = self.fc1_main(x_main)
        x_main = paddle.nn.functional.gelu(x_main)
        x_main = self.fc2_main(x_main)
        x_main = paddle.nn.functional.gelu(x_main)
        x_main = paddle.reshape(x_main, (x_main.shape[0], 32, 32, 10))

        # Branch path
        x_branch = self.branch_path(x)

        # Combine main and branch paths
        x_combined = x_main + x_branch

        # Final output
        x_final = self.final_fc1(x_combined)
        return x_final

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 10, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)