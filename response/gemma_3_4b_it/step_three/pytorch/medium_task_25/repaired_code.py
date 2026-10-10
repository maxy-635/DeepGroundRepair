import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0)
        )

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Sequential(
                nn.Conv1d(32, 16, kernel_size=1, stride=1, padding=0),
                nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
            ),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Sequential(
                nn.Conv1d(32, 16, kernel_size=1, stride=1, padding=0),
                nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0),
                nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
            )
        )

        # Concatenate the outputs of the paths
        self.concat_dim = 1
        self.concat = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0),
            torch.cat([self.path1(input_tensor), self.path2(input_tensor), self.path3(input_tensor), self.path4(input_tensor)], dim=self.concat_dim)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)

        x = torch.cat([x1, x2, x3, x4], dim=self.concat_dim)
        x = self.concat(x)
        x = F.relu(x)
        output = self.fc(x)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Batch size 1, 3 channels, 32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)