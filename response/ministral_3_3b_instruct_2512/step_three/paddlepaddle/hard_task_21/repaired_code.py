import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with three branches
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(output_size=(1, 1)),
            nn.Flatten()
        )

        # Branch path with 1x1 convolution to match main path channels
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final fully connected layers
        self.fc1 = nn.Linear(in_features=64, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process main path
        main_output = self.main_path(x)  # output tensor shape: (1, 64)

        # Process branch path
        branch_output = self.branch_path(x)  # output tensor shape: (1, 64, 32, 32)

        # Add outputs from main and branch paths
        combined_output = paddle.add(main_output, paddle.reshape(branch_output, shape=(1, -1)))

        # Reshape for fully connected layers
        combined_output = combined_output.reshape((-1, 64))

        # Fully connected layers
        combined_output = self.fc1(combined_output)
        combined_output = self.fc2(combined_output)

        return combined_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Output:", output_tensor)