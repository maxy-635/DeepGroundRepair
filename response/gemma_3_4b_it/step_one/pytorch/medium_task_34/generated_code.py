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
        self.conv_out = nn.Conv1d(32, 10, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First part
        x = self.conv1_1(x)
        x = self.pool1_1(x)
        x = self.conv1_2(x)
        x = self.pool1_2(x)
        x = self.conv1_3(x)
        x = self.pool1_3(x)

        # Second part
        x = self.conv2_1(x)
        x = self.dropout1(x)
        x = self.conv2_2(x)

        # Upsampling part
        x = self.upsample1_1(x)
        x = self.skip_conv1_3(x)
        x = self.upsample1_2(x)
        x = self.skip_conv1_2(x)
        x = self.upsample1_3(x)
        x = self.skip_conv1_1(x)

        # Final layer
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)