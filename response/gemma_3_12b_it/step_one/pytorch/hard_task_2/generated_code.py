import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        x1 = self.group1(x)
        x2 = self.group2(x)
        x3 = self.group3(x)
        main_path = torch.add(x1, x2)
        main_path = torch.add(main_path, x3)
        combined_features = torch.add(main_path, x)
        flattened = torch.flatten(combined_features, 1)
        output = self.fc(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)