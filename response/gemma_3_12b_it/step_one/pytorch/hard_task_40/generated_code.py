import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.avgpool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=13824, out_features=128)  # Assuming input image size is 28x28
        self.reshape = nn.Reshape(shape=(128, 4))

        # Second block - parallel paths
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )
        self.concat = nn.Cat(dim=1)

        # Output layers
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block
        x = self.avgpool1(x)
        x = self.avgpool2(x)
        x = self.avgpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Output layers
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)