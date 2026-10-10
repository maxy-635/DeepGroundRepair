import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.attention_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.reduction_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.layer_norm = nn.LayerNorm(normalized_shape=(1, 32, 32), eps=1e-05, elementwise_affine=True)
        self.relu = nn.ReLU(inplace=False)
        self.restore_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1, 1), stride=(1, 1))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features=3072, out_features=10, bias=True)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Generate attention weights
        attention_weights = self.softmax(self.attention_conv(x))  # output tensor shape: (1, 3, 32, 32)
        # Apply attention to input features
        attended_features = x * attention_weights  # output tensor shape: (1, 3, 32, 32)
        # Reduce dimensionality
        reduced_features = self.reduction_conv(attended_features)  # output tensor shape: (1, 1, 32, 32)
        # Apply layer normalization and ReLU
        normalized_features = self.layer_norm(reduced_features)
        activated_features = self.relu(normalized_features)
        # Restore dimensionality
        restored_features = self.restore_conv(activated_features)
        # Add to original input
        residual_features = x + restored_features
        # Flatten and classify
        flattened_features = self.flatten(residual_features)
        output = self.fc(flattened_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)