import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1 with 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        # Branch 2 with 5x5 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax(dim=1)
        )

        # Final fully connected layer
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Process both branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # Combine branches
        combined = torch.add(branch1_out, branch2_out)

        # Get attention weights
        attention_weights = self.attention(combined.squeeze(-1).squeeze(-1))

        # Apply attention weights
        weighted_branch1 = torch.mul(branch1_out.squeeze(-1).squeeze(-1), attention_weights[:, 0].unsqueeze(1))
        weighted_branch2 = torch.mul(branch2_out.squeeze(-1).squeeze(-1), attention_weights[:, 1].unsqueeze(1))

        # Combine weighted branches
        final_combined = torch.add(weighted_branch1, weighted_branch2)

        # Final classification
        output = self.fc(final_combined)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)