import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        group1_out = self.group1(x)  # output tensor shape: (1, 32, 32, 32)
        group2_out = self.group2(x)  # output tensor shape: (1, 32, 32, 32)
        group3_out = self.group3(x)  # output tensor shape: (1, 32, 32, 32)

        combined = torch.cat((group1_out, group2_out, group3_out), dim=1)  # output tensor shape: (1, 96, 32, 32)
        combined = combined.view(combined.size(0), -1)  # output tensor shape: (1, 98304)

        fc1_out = self.fc1(combined)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)

        return F.softmax(fc3_out, dim=1)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)