import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.avgpool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1024, out_features=256)  # Assuming 28x28 input
        self.reshape = nn.Reshape(shape='-1')

        # Second Block
        self.split = nn.Split(split_size=64)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.concat = nn.Sequential()
        self.flatten_final = nn.Flatten()
        self.fc2 = nn.Linear(in_features=32*64, out_features=10) #10 classes for MNIST



    def forward(self, x):
        # First Block
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second Block
        x = self.split(x)

        x1 = self.conv1(x[0])
        x2 = self.conv2(x[1])
        x3 = self.conv3(x[2])
        x4 = self.conv4(x[3])
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.flatten_final(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)