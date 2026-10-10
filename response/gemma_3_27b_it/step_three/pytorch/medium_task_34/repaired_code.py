import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature Extraction
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Generalization Enhancement
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Upsampling and Skip Connections
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv6_1 = nn.Conv2d(in_channels=64 + 128, out_channels=64, kernel_size=3, stride=1, padding=1)  # Skip connection from conv3_1

        self.conv7_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv8_1 = nn.Conv2d(in_channels=32 + 64, out_channels=32, kernel_size=3, stride=1, padding=1)  # Skip connection from conv2_1

        self.conv9_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.conv10_1 = nn.Conv2d(in_channels=16 + 32, out_channels=16, kernel_size=3, stride=1, padding=1)  # Skip connection from conv1_1

        # Classification
        self.conv11_1 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature Extraction
        x = F.relu(self.conv1_1(x))  # output tensor shape: (1, 64, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 64, 16, 16)
        x = F.relu(self.conv2_1(x))  # output tensor shape: (1, 128, 16, 16)
        x = self.pool2(x)  # output tensor shape: (1, 128, 8, 8)
        x = F.relu(self.conv3_1(x))  # output tensor shape: (1, 256, 8, 8)
        x = self.pool3(x)  # output tensor shape: (1, 256, 4, 4)

        # Generalization Enhancement
        x = F.relu(self.conv4_1(x))  # output tensor shape: (1, 256, 4, 4)
        x = self.dropout1(x)  # output tensor shape: (1, 256, 4, 4)
        x = F.relu(self.conv4_2(x))  # output tensor shape: (1, 256, 4, 4)

        # Upsampling and Skip Connections
        x = F.relu(self.conv5_1(x))  # output tensor shape: (1, 128, 4, 4)
        x = self.deconv1(x)  # output tensor shape: (1, 64, 8, 8)
        x = torch.cat([x, self.conv3_1(x)], dim=1)
        x = F.relu(self.conv6_1(x))

        x = F.relu(self.conv7_1(x))
        x = self.deconv2(x)
        x = torch.cat([x, self.conv2_1(x)], dim=1)
        x = F.relu(self.conv8_1(x))

        x = F.relu(self.conv9_1(x))
        x = self.deconv3(x)
        x = torch.cat([x, self.conv1_1(x)], dim=1)
        x = F.relu(self.conv10_1(x))

        # Classification
        x = self.conv11_1(x)
        x = self.softmax(x.view(x.size(0), -1))

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)