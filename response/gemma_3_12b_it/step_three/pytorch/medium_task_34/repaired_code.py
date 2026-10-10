import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Generalization enhancement block
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Upsampling block
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.tconv4_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.tconv5_1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.conv6_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.tconv6_1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Classification
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1_1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv2_1(x))
        x = self.pool3(x)

        # Generalization enhancement
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = F.relu(self.conv3_2(x))

        # Upsampling with skip connections
        skip_connection1 = self.conv1_1(torch.zeros_like(x))
        skip_connection2 = self.conv1_2(torch.zeros_like(x))
        skip_connection3 = self.conv2_1(torch.zeros_like(x))

        x = F.relu(self.conv4_1(x))
        x = self.tconv4_1(x)
        x = x + skip_connection3
        x = F.relu(self.conv5_1(x))
        x = self.tconv5_1(x)
        x = x + skip_connection2
        x = F.relu(self.conv6_1(x))
        x = self.tconv6_1(x)
        x = x + skip_connection1

        # Classification
        x = self.conv7(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)