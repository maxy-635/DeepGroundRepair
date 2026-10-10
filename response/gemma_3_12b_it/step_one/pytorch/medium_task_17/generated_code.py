import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.linear = nn.Linear(in_features=1024, out_features=10)  # Assuming flattened input size of 1024, 10 classes

    def forward(self, x):
        # Get the shape of the input layer
        batch_size, channels, height, width = x.shape

        # Reshape the input tensor into groups
        groups = 3
        channels_per_group = channels // groups
        x = torch.reshape(x, (batch_size, height, width, groups, channels_per_group))

        # Channel shuffling using permutation
        dims = (0, 2, 3, 1, 4)  # Rearrange dimensions to enable channel shuffling
        x = torch.permute(x, dims)

        # Reshape back to the original input shape
        x = torch.reshape(x, (batch_size, channels, height, width))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layer with softmax activation
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, channels, height, width)
    batch_size = 1
    channels = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)