import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=16 * 32 * 32, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x1 = self.model[1](x)
        x2 = self.model[2](x)
        x = torch.concatenate([x1, x2], dim=1)
        x = self.model[3](x)
        x = self.model[4](x)
        x = self.model[5](x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)