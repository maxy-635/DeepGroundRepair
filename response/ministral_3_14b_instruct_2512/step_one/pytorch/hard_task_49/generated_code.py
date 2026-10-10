import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three average pooling layers with different scales
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate the pooled features
        self.flatten = nn.Flatten()

        # Fully connected layer and reshape for second block
        self.fc1 = nn.Linear(3*28*28, 28*28*4)  # MNIST images are 28x28
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (1, 28, 28, 4))  # Reshape to (batch_size, 1, 28, 28, 4)
        )

        # Second block: depthwise separable convolutions with different kernel sizes
        self.depthwise_conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, groups=1),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        )
        self.depthwise_conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        )
        self.depthwise_conv3 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, groups=1),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        )
        self.depthwise_conv4 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, groups=1),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        )

        # Final classification layer
        self.fc2 = nn.Linear(4*28*28, 10)  # 10 classes for MNIST

    def forward(self, x):
        # First block processing
        pool1 = self.avg_pool1(x)
        pool2 = self.avg_pool2(x)
        pool3 = self.avg_pool3(x)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)
        pool2_flat = self.flatten(pool2)
        pool3_flat = self.flatten(pool3)

        # Concatenate along channel dimension
        concatenated = torch.cat([pool1_flat, pool2_flat, pool3_flat], dim=1)

        # Transform for second block
        transformed = self.fc1(concatenated)
        reshaped = self.reshape(transformed)

        # Second block processing
        # Split into 4 groups along last dimension
        group1 = reshaped[:, :, :, :, 0:1]
        group2 = reshaped[:, :, :, :, 1:2]
        group3 = reshaped[:, :, :, :, 2:3]
        group4 = reshaped[:, :, :, :, 3:4]

        # Process each group with different depthwise convolutions
        conv1 = self.depthwise_conv1(group1)
        conv2 = self.depthwise_conv2(group2)
        conv3 = self.depthwise_conv3(group3)
        conv4 = self.depthwise_conv4(group4)

        # Concatenate the results
        concatenated_conv = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        # Final classification
        flattened = self.flatten(concatenated_conv)
        output = self.fc2(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # MNIST images are 28x28 with 1 channel
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)