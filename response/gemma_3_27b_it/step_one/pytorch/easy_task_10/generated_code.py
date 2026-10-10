import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, groups=8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=16 * 7 * 7, out_features=10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)