import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split into three channel groups and apply separable convolutions
        self.conv_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.conv_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features='mask', out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three channel groups
        group1 = self.conv_group1(x)  # output tensor shape: (1, 32, 16, 16)
        group2 = self.conv_group2(x)  # output tensor shape: (1, 32, 16, 16)
        group3 = self.conv_group3(x)  # output tensor shape: (1, 32, 16, 16)

        # Concatenate the outputs
        combined = torch.cat((group1, group2, group3), dim=1)  # output tensor shape: (1, 96, 16, 16)
        combined = combined.view(combined.size(0), -1)  # output tensor shape: (1, 24576)
        output = self.concat(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)