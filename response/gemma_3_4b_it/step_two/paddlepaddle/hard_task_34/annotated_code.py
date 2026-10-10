import sys
import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fusion
        self.concat = nn.Concat(dim='mask')
        self.add = nn.Add()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Adjust output size based on MNIST

    def forward(self, x):
        # Main path
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        # Branch path
        x_branch = self.conv_branch(x)

        # Fusion
        x = self.concat(x, x_branch)
        x = self.add(x, x)

        # Classification
        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)