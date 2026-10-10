import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Reshape input tensor into (height, width, groups, channels_per_group)
        self.reshape_layer = nn.Unflatten(3, (3, 32))  # Assuming input shape is (batch_size, 32, 32, 3)

        # Swap dimensions 3 and 4 (channel shuffling)
        self.permute_layer = nn.Permute('mask')

        # Reshape back to original input shape
        self.reshape_back_layer = nn.Unflatten(3, (32, 32, 3))

        # Fully connected layer with softmax activation
        self.fully_connected = nn.Sequential(
            nn.Linear(32 * 32 * 3, 10),  # Assuming input is flattened to 32*32*3
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Reshape input tensor
        x = self.reshape_layer(x)

        # Swap dimensions
        x = self.permute_layer(x)

        # Reshape back to original shape
        x = self.reshape_back_layer(x)

        # Flatten and pass through fully connected layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fully_connected(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, 32, 32, 3)
    input_tensor = torch.randn(4, 32, 32, 3)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Output values:", output_tensor)