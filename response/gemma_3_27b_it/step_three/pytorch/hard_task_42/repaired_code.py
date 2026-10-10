import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_path1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2),
            nn.Flatten()
        )
        self.block1_path2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.2),
            nn.Flatten()
        )
        self.block1_path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Dropout(p=0.2),
            nn.Flatten()
        )

        # Fully connected layer and reshaping
        self.fc1 = nn.Linear(in_features=3*784, out_features=128)
        self.reshape = nn.Reshape(shape=(1, 128))

        # Block 2
        self.block2_path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_path2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=7, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=0)
        )
        self.block2_path3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=7, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=7, padding=0)
        )
        self.block2_path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers for classification
        self.fc2 = nn.Linear(in_features=32*4, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        out1_path1 = self.block1_path1(x)
        out1_path2 = self.block1_path2(x)
        out1_path3 = self.block1_path3(x)
        out1 = torch.cat([out1_path1, out1_path2, out1_path3], dim=1)

        # Fully connected layer and reshaping
        out2 = self.fc1(out1)
        out2 = self.reshape(out2)

        # Block 2
        out2_path1 = self.block2_path1(out2)
        out2_path2 = self.block2_path2(out2)
        out2_path3 = self.block2_path3(out2)
        out2_path4 = self.block2_path4(out2)
        out2 = torch.cat([out2_path1, out2_path2, out2_path3, out2_path4], dim=1)
        out2 = out2.view(out2.size(0), -1)  # Flatten for FC layers

        # Fully connected layers for classification
        out3 = F.relu(self.fc2(out2))
        out4 = self.fc3(out3)

        return out4


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)