import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 5x5 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
            nn.Softmax(dim=1)
        )

        # Final fully connected layer
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Process both branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # Combine features through addition
        combined = branch1_out + branch2_out

        # Apply global average pooling
        pooled = self.pool(combined)

        # Flatten for attention mechanism
        pooled_flat = pooled.view(pooled.size(0), -1)

        # Generate attention weights
        attention_weights = self.attention(pooled_flat)

        # Apply attention to each branch's output
        branch1_attended = torch.mul(branch1_out, attention_weights[:, 0:1, None, None])
        branch2_attended = torch.mul(branch2_out, attention_weights[:, 1:2, None, None])

        # Combine attended features
        attended_combined = branch1_attended + branch2_attended

        # Apply global average pooling again
        attended_pooled = self.pool(attended_combined)

        # Flatten and pass through final FC layer
        output = self.fc(attended_pooled.view(attended_pooled.size(0), -1))

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)