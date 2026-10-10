import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.fc = nn.Linear(in_features=3072, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (2, 3, 32, 32)
        # Get the shape of the input tensor
        input_shape = x.shape  # output tensor shape: (unknown)

        # Reshape the input tensor into three groups
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]  # output tensor shape: (1); (1); (1)
        groups = 3  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)
        reshaped = torch.reshape('mask', 'mask')

        # Swap the third and fourth dimensions for channel shuffling
        permuted = torch.permute(reshaped, (0, 1, 2, 4, 3))

        # Reshape back to the original input shape
        reshaped_back = torch.reshape(permuted, input_shape)

        # Flatten the tensor for the fully connected layer
        flattened = torch.flatten(reshaped_back, start_dim=1)

        # Pass through the fully connected layer with softmax activation
        output = torch.softmax(self.fc(flattened), dim=1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(2, 3, 32, 32)  # Batch size of 2, 3 channels, 32x32 images
    output_tensor = model(input_tensor)
    print(output_tensor)