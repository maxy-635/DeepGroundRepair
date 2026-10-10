import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 4, out_features=10)

    def forward(self, x):
        # Process each path
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)

        # Concatenate the outputs of all paths
        out = torch.cat((out1, out2, out3, out4), dim=1)

        # Global average pooling to reduce spatial dimensions
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)

        # Classification
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])