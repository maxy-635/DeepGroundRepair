import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.reduction_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(normalized_shape='mask')
        self.restore_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3 * 3 * 3, out_features=10)  # Assuming CIFAR-10, 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Attention mechanism
        attention_weights = self.attention_conv(x)  # output tensor shape: (1, 1, 32, 32)
        attention_weights = self.softmax(attention_weights)  # output tensor shape: (1, 1, 32, 32)
        weighted_features = torch.mul(x, attention_weights)  # output tensor shape: (1, 3, 32, 32)

        # Dimensionality reduction
        reduced_features = self.reduction_conv(x)  # output tensor shape: (1, 1, 32, 32)
        reduced_features = self.layer_norm(reduced_features)
        reduced_features = F.relu(reduced_features)
        restored_features = self.restore_conv(reduced_features)

        # Residual connection
        x = torch.add(x, restored_features)

        # Classification
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)