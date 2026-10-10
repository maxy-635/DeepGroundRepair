import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.linear = nn.Linear(in_features=32 * 3, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 32, 3)
        # Get the shape of the input tensor
        height, width, channels = x.shape  # output tensor shape: (1); (1); (1)

        # Reshape the input tensor into three groups
        groups = 3  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)
        x = torch.reshape(x, (height, width, groups, channels_per_group))  # output tensor shape: (32, 32, 3, 1)

        # Swap the third and fourth dimensions for channel shuffling
        x = torch.permute(x, (0, 1, 3, 2))  # output tensor shape: (32, 32, 1, 3)

        # Reshape back to the original input shape
        x = torch.reshape(x, (height, width, channels))  # output tensor shape: (32, 32, 3)

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