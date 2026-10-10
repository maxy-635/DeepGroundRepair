import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Part 1: Feature extraction with 3 pairs of <convolutional layer, max-pooling layer>
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Part 2: Enhance generalization with <convolutional layer, Dropout layer, convolutional layer>
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        # Part 3: Upsampling with 3 pairs of <convolutional layer, transposed convolutional layer>
        self.upconv1 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)

        # Skip connections
        self.skip_conv1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask')
        self.skip_conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.skip_conv3 = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        # Final classification layer
        self.final_conv = nn.Conv2d(128, 10, kernel_size=1, stride=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Part 1: Feature extraction
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 16, 16)

        x2 = self.conv2(x1)  # output tensor shape: (1, 64, 16, 16)
        x2 = self.pool2(x2)  # output tensor shape: (1, 64, 8, 8)

        x3 = self.conv3(x2)  # output tensor shape: (1, 128, 8, 8)
        x3 = self.pool3(x3)  # output tensor shape: (1, 128, 4, 4)

        # Part 2: Enhance generalization
        x4 = self.conv4(x3)  # output tensor shape: (1, 64, 4, 4)
        x4 = self.dropout(x4)  # output tensor shape: (1, 64, 4, 4)
        x4 = self.conv5(x4)  # output tensor shape: (1, 32, 4, 4)

        # Part 3: Upsampling with skip connections
        x_up1 = self.upconv1(x4)  # output tensor shape: (1, 64, 8, 8)
        x_up1 = torch.cat([x_up1, self.skip_conv1(x3)], dim=1)

        x_up2 = self.upconv2(x_up1)
        x_up2 = torch.cat([x_up2, self.skip_conv2(x2)], dim=1)

        x_up3 = self.upconv3(x_up2)
        x_up3 = torch.cat([x_up3, self.skip_conv3(x1)], dim=1)

        # Final classification
        x_final = self.final_conv(x_up3)
        return x_final

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)