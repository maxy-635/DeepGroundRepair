import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: Feature extraction
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second part: Enhancement
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Upsampling part
        self.upsample1_1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.skip_conv1_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.upsample1_2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.skip_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample1_3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.skip_conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Final layer
        self.conv_out = nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First part
        x = self.conv1_1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.pool1_1(x)  # output tensor shape: (1, 32, 16, 16)
        x = self.conv1_2(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.pool1_2(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.conv1_3(x)  # output tensor shape: (1, 128, 8, 8)
        x = self.pool1_3(x)  # output tensor shape: (1, 128, 4, 4)

        # Second part
        x = self.conv2_1(x)  # output tensor shape: (1, 128, 4, 4)
        x = self.dropout1(x)  # output tensor shape: (1, 128, 4, 4)
        x = self.conv2_2(x)  # output tensor shape: (1, 256, 4, 4)

        # Upsampling part
        x = self.upsample1_1(x)  # output tensor shape: (1, 128, 8, 8)
        x = self.skip_conv1_3(x)  # output tensor shape: (1, 128, 8, 8)
        x = self.upsample1_2(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.skip_conv1_2(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.upsample1_3(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.skip_conv1_1(x)  # output tensor shape: (1, 32, 32, 32)

        # Final layer
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)