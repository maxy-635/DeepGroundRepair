import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = nn.Sequential(
            nn.DepthwiseConv2d(in_channels=1, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.branch2 = nn.Sequential(
            nn.DepthwiseConv2d(in_channels=1, kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.branch3 = nn.Sequential(
            nn.DepthwiseConv2d(in_channels=1, kernel_size=7, stride=1, padding=3, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 + 64 + 128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        concatenated = torch.cat((out1, out2, out3), dim=1)
        output = self.concat(concatenated)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)