import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction part
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Generalization enhancement part
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Upsampling part
        self.upconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.upconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Final classification layer
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature extraction
        x1 = self.pool1(nn.functional.relu(self.conv1(x)))  # output tensor shape: (1, 64, 16, 16)
        x2 = self.pool2(nn.functional.relu(self.conv2(x1)))  # output tensor shape: (1, 128, 8, 8)
        x3 = self.pool3(nn.functional.relu(self.conv3(x2)))  # output tensor shape: (1, 256, 4, 4)

        # Generalization enhancement
        x4 = nn.functional.relu(self.conv4(x3))  # output tensor shape: (1, 256, 4, 4)
        x4 = self.dropout(x4)  # output tensor shape: (1, 256, 4, 4)
        x5 = nn.functional.relu(self.conv5(x4))  # output tensor shape: (1, 256, 4, 4)

        # Upsampling with skip connections
        x6 = self.upconv1(x5)  # output tensor shape: (1, 128, 8, 8)
        x6 = torch.cat([x6, x2], dim=1)  # output tensor shape: (1, 256, 8, 8)
        x6 = nn.functional.relu(self.conv6(x6))  # output tensor shape: (1, 128, 8, 8)

        x7 = self.upconv2(x6)  # output tensor shape: (1, 64, 16, 16)
        x7 = torch.cat([x7, x1], dim=1)  # output tensor shape: (1, 128, 16, 16)
        x7 = nn.functional.relu(self.conv7(x7))  # output tensor shape: (1, 64, 16, 16)

        x8 = self.upconv3(x7)  # output tensor shape: (1, 32, 32, 32)
        x8 = torch.cat([x8, x], dim=1)  # output tensor shape: (1, 35, 32, 32)
        x8 = nn.functional.relu(self.conv8(x8))

        # Final classification
        x9 = self.final_conv(x8)
        x9 = self.softmax(x9)

        return x9

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)