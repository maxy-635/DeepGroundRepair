import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three average pooling layers with specified windows and strides
        self.pool1 = nn.AdaptiveAvgPool2d(output_size=(None, 2))
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(None, 2))
        self.pool3 = nn.AdaptiveAvgPool2d(output_size=(None, 4))

        # Define the convolutional layers to extract features before pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Flatten layers to concatenate outputs of pooling layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate the flattened outputs of the pooling layers
        self.concatenate = nn.Sequential(
            nn.Linear(in_features='mask', out_features=256),
            nn.Linear(256, 512)   # Adjust dimensions as needed based on actual pooled output sizes
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply convolutional layers
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.conv3(x)  # output tensor shape: (1, 128, 32, 32)

        # Apply pooling layers
        x1 = self.pool1(x)  # output tensor shape: (1, 128, 32, 2)
        x2 = self.pool2(x)  # output tensor shape: (1, 128, 32, 2)
        x3 = self.pool3(x)  # output tensor shape: (1, 128, 32, 4)

        # Flatten the pooled outputs
        x1 = self.flatten1(x1)  # output tensor shape: (1, 8192)
        x2 = self.flatten2(x2)  # output tensor shape: (1, 8192)
        x3 = self.flatten3(x3)  # output tensor shape: (1, 16384)

        # Concatenate the flattened outputs
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 32768)

        # Pass through fully connected layers
        x = self.concatenate(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input for CIFAR-10 (32x32 images)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)