import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Stage 1: Downsampling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: Downsampling
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional and Dropout Layers
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # Upsampling and Skip Connections
        self.upsample1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128 + 128, out_channels=128, kernel_size=3, stride=1, padding=1)  # Skip connection

        self.upsample2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)  # Skip connection

        # Output Layer
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Downsampling
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 64, 16, 16)

        x = self.conv2(x)  # output tensor shape: (1, 128, 16, 16)
        x = self.relu2(x)  # output tensor shape: (1, 128, 16, 16)
        x = self.pool2(x)  # output tensor shape: (1, 128, 8, 8)

        # Convolutional and Dropout
        x = self.conv3(x)  # output tensor shape: (1, 256, 8, 8)
        x = self.relu3(x)  # output tensor shape: (1, 256, 8, 8)
        x = self.dropout(x)  # output tensor shape: (1, 256, 8, 8)

        # Upsampling and Skip Connections
        x = self.upsample1(x)  # output tensor shape: (1, 128, 16, 16)
        x = torch.cat([x, self.pool2(x)], dim=1)
        x = self.conv4(x)
        x = self.relu4(x)

        x = self.upsample2(x)
        x = torch.cat([x, self.pool1(x)], dim=1)
        x = self.conv5(x)
        x = self.relu5(x)

        # Output
        x = self.conv6(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10, 32, 32])