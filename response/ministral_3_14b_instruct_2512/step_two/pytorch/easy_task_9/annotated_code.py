import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.ReLU(inplace=True)
        )
        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # 1x1 conv to increase dimensionality
        x_original = x
        x = self.conv1x1_1(x)

        # 3x3 depthwise separable conv
        x = self.depthwise_conv(x)

        # 1x1 conv to reduce dimensionality
        x = self.conv1x1_2(x)

        # Add original input to processed output
        x = torch.add(x, x_original)

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)