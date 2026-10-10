import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2DTranspose(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Additional layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.conv1(x)
        x_main = self.conv2(x_main)

        # Branch path
        x_branch = self.branch_path(x)

        # Combine paths
        x_combined = paddle.add(x_main, x_branch)

        # Flatten and fully connected layer
        x_flattened = self.flatten(x_combined)
        output = self.fc(x_flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)