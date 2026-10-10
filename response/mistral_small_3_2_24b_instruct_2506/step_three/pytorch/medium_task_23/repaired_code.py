import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 4 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # Process each path
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3, out4), dim=1)

        # Flatten and pass through the fully connected layer
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)