import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention mechanism components
        self.attention_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.attention_softmax = nn.Softmax(dim=1)

        # Dimensionality reduction components
        self.reduce_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(normalized_shape=(32, 32), eps=1e-5, elementwise_affine=True)
        self.relu = nn.ReLU(inplace=False)

        # Dimensionality restoration components
        self.restore_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Classification components
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Generate attention weights
        attention_weights = self.attention_conv(x)  # output tensor shape: (1, 3, 32, 32)
        attention_weights = self.attention_softmax(attention_weights)  # output tensor shape: (1, 3, 32, 32)

        # Apply attention to input features
        attended_features = attention_weights * x  # output tensor shape: (1, 3, 32, 32)

        # Reduce dimensionality
        reduced_features = self.reduce_conv(attended_features)  # output tensor shape: (1, 1, 32, 32)
        reduced_features = self.layer_norm(reduced_features)
        reduced_features = self.relu(reduced_features)

        # Restore dimensionality
        restored_features = self.restore_conv(reduced_features)

        # Add to original input
        residual_features = x + restored_features

        # Classification
        flattened = self.flatten(residual_features)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)