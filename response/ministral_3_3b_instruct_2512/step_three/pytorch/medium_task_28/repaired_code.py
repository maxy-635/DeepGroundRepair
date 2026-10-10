import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention mechanism with 1x1 convolution and softmax
        self.attention = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        # Feature reduction and restoration
        self.reduce = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=(32*32*3,)),
            nn.ReLU(inplace=True)
        )

        self.restore = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Generate attention weights
        attention_weights = self.attention(x)  # output tensor shape: (1, 3, 32, 32)

        # Weighted processing
        weighted_features = torch.add(x, torch.mul(x, attention_weights))  # output tensor shape: (1, 3, 32, 32)

        # Feature reduction and restoration
        reduced_features = self.reduce(weighted_features)
        restored_features = self.restore(reduced_features)

        # Add original input to restored features
        combined_features = torch.add(weighted_features, restored_features)

        # Classification
        output = self.classifier(combined_features)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)

    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)