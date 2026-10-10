import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional Branch 1 (3x3 kernel)
        self.conv1_branch1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_branch1 = nn.ReLU(inplace=True)
        self.conv2_branch1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2_branch1 = nn.ReLU(inplace=True)

        # Convolutional Branch 2 (5x5 kernel)
        self.conv1_branch2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu1_branch2 = nn.ReLU(inplace=True)
        self.conv2_branch2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2_branch2 = nn.ReLU(inplace=True)

        # Feature Combination
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1_attention = nn.Linear(in_features=64, out_features=16)
        self.relu_attention = nn.ReLU(inplace=True)
        self.fc2_attention = nn.Linear(in_features=16, out_features=2)
        self.softmax_attention = nn.Softmax(dim=1)

        # Classification
        self.fc_classifier = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Branch 1
        branch1_output = self.relu2_branch1(self.conv2_branch1(self.relu1_branch1(self.conv1_branch1(x))))  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2_output = self.relu2_branch2(self.conv2_branch2(self.relu1_branch2(self.conv1_branch2(x))))  # output tensor shape: (1, 64, 32, 32)

        # Feature Combination
        combined_features = branch1_output + branch2_output  # output tensor shape: (1, 64, 32, 32)
        pooled_features = self.adaptive_pool(combined_features)  # output tensor shape: (1, 64, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # output tensor shape: (1, 64)

        # Attention Mechanism
        attention_output = self.relu_attention(self.fc1_attention(pooled_features))  # output tensor shape: (1, 16)
        attention_weights = self.softmax_attention(self.fc2_attention(attention_output))  # output tensor shape: (1, 2)

        # Weighted Output
        weighted_branch1_output = branch1_output  # output tensor shape: (1, 64, 32, 32)
        weighted_branch2_output = branch2_output  # output tensor shape: (1, 64, 32, 32)
        final_weighted_output = attention_weights[:, 0] * weighted_branch1_output + attention_weights[:, 1] * weighted_branch2_output  # output tensor shape: (1, 64, 32, 32)
        final_weighted_output = final_weighted_output.view(final_weighted_output.size(0), -1)  # output tensor shape: (1, 65536)

        # Classification
        output = self.fc_classifier(final_weighted_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)