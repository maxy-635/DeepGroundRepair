import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64*4*32*32, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Process each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3, out4), dim=1)

        # Flatten the concatenated output
        out = out.view(out.size(0), -1)

        # Pass through fully connected layers
        out = self.fc_layers(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)