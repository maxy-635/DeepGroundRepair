import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.concat1 = nn.Concat(dim=1)

        # Block 2: Feature Extraction Branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.concat2 = nn.Concat([self.branch1.output, self.branch2.output, self.branch3.output], dim=1)

        # Global Average Pooling and Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)  # Output for CIFAR-10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.concat1

        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.concat2

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Example Usage
    batch_size = 4
    img_height = 32
    img_width = 32
    model = DL_Model()
    input_tensor = torch.randn(batch_size, 3, img_height, img_width)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)