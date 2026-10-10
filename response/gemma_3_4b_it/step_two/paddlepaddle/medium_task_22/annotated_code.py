import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms
import numpy as np

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 conv followed by two 3x3 conv
        self.branch2 = nn.Conv1D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Branch 3: Max pooling
        self.branch3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Multi-scale feature fusion
        self.merge = nn.Conv2D(in_channels=64 + 32 + 32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Branch 1
        x1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        x2 = self.branch2(x)

        # Branch 3
        x3 = self.branch3(x)

        # Merge
        merged = self.merge(torch.cat([x1, x2, x3], dim=1))

        # Flatten
        flattened = self.flatten(merged)

        # Fully connected layers
        x = self.fc1(flattened)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.to_tensor(np.random.rand(1, 3, 32, 32), dtype=paddle.float32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)