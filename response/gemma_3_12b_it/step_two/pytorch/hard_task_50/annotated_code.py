import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.reshape = nn.Reshape(shape='mask')

        # Second Block
        self.split = nn.Split(split_size=1)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.concat = nn.Concatenate(dim=1)
        self.fc2 = nn.Linear(in_features=256, out_features=10)


    def forward(self, x):
        # First Block
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x)
        x3 = self.maxpool3(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)

        x = self.dropout(torch.cat((x1, x2, x3), dim=1))
        x = self.fc1(x)
        x = self.reshape(x)

        # Second Block
        x = self.split(x)

        x1 = self.conv1(x[0])
        x2 = self.conv2(x[0])
        x3 = self.conv3(x[0])
        x4 = self.conv4(x[0])

        x = self.concat((x1, x2, x3, x4))
        x = x.view(x.size(0), -1)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 3, 32, 32)  # Example input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)