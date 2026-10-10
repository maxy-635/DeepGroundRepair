import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Depthwise separable convolutions
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.concat1 = nn.Concat([self.conv1.output, self.conv2.output, self.conv3.output])

        # Second block: Feature extraction branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.concat2 = nn.Concat([self.branch1.output, self.branch2.output, self.branch3.output, self.branch4.output, self.branch5.output])

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(concat2.shape[1] * concat2.shape[2] * concat2.shape[3], 10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu2(x3)
        x5 = self.conv3(x4)
        x6 = self.relu3(x5)
        x = self.concat1(x6)

        x7 = self.branch1(x)
        x8 = self.branch2(x)
        x9 = self.branch3(x)
        x10 = self.branch4(x)
        x11 = self.branch5(x)
        x = self.concat2(x11)

        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)