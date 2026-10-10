import torch
import torch.nn as nn
import torch.utils.data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Concatenate branches
        self.concat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolutional layer
        x = self.conv1(x)

        # Branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Concatenate branches
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.concat(x)

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (3x32x32)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)