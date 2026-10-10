import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        self.concat1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True)
        )

        # Second block
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=0)

        self.concat2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x = self.avg_pool3(x)  # output tensor shape: (1, 1, 3, 3)
        x = self.concat1(x)

        # Second block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.concat2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)