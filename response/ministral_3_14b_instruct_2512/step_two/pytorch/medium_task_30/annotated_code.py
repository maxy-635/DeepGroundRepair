import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define three average pooling layers with different kernel sizes and strides
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Flatten layers for each pooling output
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(512, 10)    # Output layer for 10 classes in CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply pooling layers
        x1 = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.pool2(x)  # output tensor shape: (1, 3, 16, 16)
        x3 = self.pool3(x)  # output tensor shape: (1, 3, 8, 8)

        # Flatten each pooling output
        x1_flat = self.flatten1(x1)  # output tensor shape: (1, 3072)
        x2_flat = self.flatten2(x2)  # output tensor shape: (1, 768)
        x3_flat = self.flatten3(x3)  # output tensor shape: (1, 192)

        # Concatenate the flattened features
        concatenated = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)  # output tensor shape: (1, 4032)

        # Process through fully connected layers
        x = self.fc1(concatenated)
        x = F.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, typical input shape is (3, 32, 32)
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)