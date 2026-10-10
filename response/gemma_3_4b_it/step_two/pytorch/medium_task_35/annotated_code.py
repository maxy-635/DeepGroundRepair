import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layers for downsampling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional layers with dropout
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.25)

        # Upsampling layers with skip connections
        self.upsample1 = nn.Upsample(size='mask', scale_factor='mask', mode='mask', align_corners='mask')
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.skip_connection1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.upsample2 = nn.Upsample(size=(16, 16), scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.skip_connection2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        # Final convolutional layer and sigmoid activation
        self.conv_out = nn.Conv2d(16, 10, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Downsampling
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 16, 16, 16)
        x = self.conv2(x)  # output tensor shape: (1, 32, 16, 16)
        x = self.pool2(x)  # output tensor shape: (1, 32, 8, 8)

        # Convolutional layers with dropout
        x = self.conv3(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.dropout(x)  # output tensor shape: (1, 64, 8, 8)

        # Upsampling with skip connections
        x = self.upsample1(x)
        x = torch.cat([x, self.skip_connection1(x)], dim=1)
        x = self.conv4(x)

        x = self.upsample2(x)
        x = torch.cat([x, self.skip_connection2(x)], dim=1)
        x = self.conv5(x)

        # Final output layer
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    # Define the input tensor shape
    input_shape = (3, 32, 32)  # Channels, Height, Width
    dummy_input = torch.randn(1, *input_shape)  # Batch size of 1

    model = DL_Model()
    output_tensor = model(dummy_input)
    print("Output tensor shape:", output_tensor.shape)