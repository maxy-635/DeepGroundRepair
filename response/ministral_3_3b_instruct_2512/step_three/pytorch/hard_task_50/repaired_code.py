import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three max pooling layers with different scales
        self.pooling_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        )

        # Flatten and dropout layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer and reshape
        self.fc = nn.Linear(in_features=3 * 32 * 32, out_features=4 * 32 * 32)
        self.reshape = nn.Reshape((3 * 32 * 32, 4 * 32 * 32))

        # Second block: separable convolutions with different kernel sizes
        self.separable_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 7))
        )

    def forward(self, x):
        # First block processing
        pooled_outputs = []
        for pool in self.pooling_block:
            pooled_outputs.append(self.flatten(pool(x)))

        # Concatenate and apply dropout
        concatenated = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(concatenated)

        # Reshape for second block
        x = self.reshape(x)

        # Second block processing
        x = self.separable_conv_block(x)

        # Flatten and pass through fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, channels=3, height=32, width=32
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)