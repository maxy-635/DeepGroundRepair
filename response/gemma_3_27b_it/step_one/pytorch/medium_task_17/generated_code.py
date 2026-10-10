import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.linear = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Get the shape of the input tensor
        height, width, channels = x.shape

        # Reshape the input tensor into three groups
        groups = 3
        channels_per_group = channels // groups
        x = torch.reshape(x, (height, width, groups, channels_per_group))

        # Swap the third and fourth dimensions for channel shuffling
        x = torch.permute(x, (0, 1, 3, 2))

        # Reshape back to the original input shape
        x = torch.reshape(x, (height, width, channels))

        # Pass through a fully connected layer with softmax activation
        x = self.linear(x.view(height * width, channels))
        output = F.softmax(x, dim=1)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with the expected shape (32, 32, 3)
    input_tensor = torch.randn(32, 32, 3)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)