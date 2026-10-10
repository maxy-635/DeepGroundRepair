import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution followed by 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 2: 1x1 convolution followed by 1x7, 7x1, and 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features='mask', out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each branch
        out1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        out3 = self.branch3(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate the outputs from all branches
        out = torch.cat((out1, out2, out3), dim=1)  # output tensor shape: (1, 131, 32, 32)

        # Flatten the concatenated output
        out = out.view(out.size(0), -1)  # output tensor shape: (1, 134144)

        # Pass through fully connected layers
        out = self.fc_layers(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])