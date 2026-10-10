import sys
import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten1 = nn.Flatten()

        # Fully connected layers for Block 1
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Block 2
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten1(x)

        # Fully connected layers for Block 1
        x = self.fc1(x)
        x = self.fc2(x)

        # Block 2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # Branch from Block 1
        branch = x

        # Fuse the main path and the branch
        combined = paddle.add(x, branch)

        # Classification
        output = self.fc2(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)