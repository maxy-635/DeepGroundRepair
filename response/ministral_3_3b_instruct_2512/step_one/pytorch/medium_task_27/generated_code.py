import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutional branch
        self.conv1_3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch 2: 5x5 convolutional branch
        self.conv2_5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global average pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),  # Assuming input size after pooling is 4x4
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes

        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Forward pass for both branches
        branch1_output = self.conv1_3x3(x)
        branch2_output = self.conv2_5x5(x)

        # Add outputs of both branches
        combined_output = branch1_output + branch2_output

        # Global average pooling
        pooled_output = self.avg_pool(combined_output)

        # Flatten the output for fully connected layers
        pooled_output = pooled_output.view(pooled_output.size(0), -1)

        # Apply attention weights
        attention_weights = self.attention_weights(pooled_output)
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), pooled_output.unsqueeze(2)).squeeze(2)

        # Final classification layer
        output = self.fc2(weighted_output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, height and width are 32, so we'll use a batch size of 1
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)