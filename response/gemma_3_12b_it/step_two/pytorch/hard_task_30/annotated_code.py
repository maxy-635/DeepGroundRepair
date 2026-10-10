import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, groups=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, groups=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        main_path = self.block1(x)
        branch_path = self.branch_path(x)
        merged = main_path + branch_path

        block2_output = self.block2(merged)
        concatenated = torch.cat([block2_output[:, :16, :, :],
                                  block2_output[:, 16:32, :, :],
                                  block2_output[:, 32:48, :, :]], dim=1)

        concat_output = self.concat(concatenated)
        flattened = torch.flatten(concat_output, 1)
        fc1_output = F.relu(self.fc1(flattened))
        output = self.fc2(fc1_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)