import paddle
import paddle.nn as nn
import numpy as np

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Conv2d(3, 32, kernel_size=1, padding=0)

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=1)
        )

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Concatenate(dim=1)
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Concatenate(dim=1)
        )

        # Feature fusion
        self.concat = nn.Concatenate(dim=1)

        # Fully connected layer for classification
        self.fc = nn.Linear(32 * 4, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)

        output = self.concat([path1, path2, path3, path4])
        output = self.fc(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)