import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Depthwise Separable Convolution and Layer Normalization
        self.conv_ds = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=3)
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32], eps=1e-05, elementwise_affine=True)

        # Fully Connected Layers for Channel-wise Transformation
        self.fc1 = nn.Linear(in_features=3, out_features=3)
        self.fc2 = nn.Linear(in_features=3, out_features=3)

        # Final Fully Connected Layers for Classification
        self.fc3 = nn.Linear(in_features=3, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Extract spatial features with depthwise separable convolution
        x = self.conv_ds(x)  # output tensor shape: (1, 3, 32, 32)
        x = self.layer_norm(x)

        # Channel-wise feature transformation with fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # Combine original input with processed features
        x = x + x

        # Classification with final fully connected layers
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image size is 32x32 with 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: torch.Size([1, 10])