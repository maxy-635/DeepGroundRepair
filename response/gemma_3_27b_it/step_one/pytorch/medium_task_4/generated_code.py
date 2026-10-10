import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Path 2
        path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        )

        # Combine paths
        self.model = nn.Sequential(
            path1,
            path2,
            nn.Flatten(),
            nn.Linear(in_features=32*32*32 + 8*32*32, out_features=10)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)