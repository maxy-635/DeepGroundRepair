import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=3*3*3, out_features=128)  # Assuming input size is 32x32x3 (CIFAR-10)
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=4*4*4*3)  # Reshape to 4D tensor
        )

        # Second block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)

        # Final classification layer
        self.fc2 = nn.Linear(in_features=64*4*4*4, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1 = self.maxpool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.maxpool2(x)  # output tensor shape: (1, 3, 16, 16)
        x3 = self.maxpool4(x)  # output tensor shape: (1, 3, 8, 8)

        x1 = torch.flatten(x1)  # output tensor shape: (3072)
        x2 = torch.flatten(x2)  # output tensor shape: (768)
        x3 = torch.flatten(x3)  # output tensor shape: (192)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.reshape(x)
        x = torch.reshape(x, (-1, 4, 4, 4, 3))

        # Second block
        x = torch.chunk(x, chunks=4, dim=-1)
        x = [self.conv1(chunk) for chunk in x]
        x = torch.cat(x, dim=1)

        # Final classification
        x = torch.flatten(x, 1)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])