import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.main_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Increase feature map dimensions
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)   # Restore number of channels
        )
        self.branch_path = nn.Conv2d(1, 1, kernel_size=1, stride=1)  # Branch path directly to input

        # Second block: three max pooling layers with different scales
        self.pooling_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),  # Assuming the flattened output size is 1024
            nn.Linear(128, 10)     # MNIST has 10 classes
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block: main path and branch path
        main_output = self.main_path(x)  # output tensor shape: (1, 1, 28, 28)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 1, 28, 28)
        combined_output = torch.add(main_output, branch_output)  # output tensor shape: (1, 1, 28, 28)

        # Second block: three max pooling layers
        pooled_outputs = self.pooling_layers(combined_output)  # output tensor shape: (1, 1, 3, 3)
        # Flatten the pooled outputs (assuming pooling layers output a single tensor)
        # Since pooling layers are sequential, we need to handle them differently
        # Here, we'll flatten each pooled output separately and concatenate them
        pooled_outputs_flattened = []  # output tensor shape: (unknown)
        for layer in self.pooling_layers:  # output tensor shape: (unknown)
            pooled_outputs_flattened.append(layer(combined_output).flatten())

        # Concatenate the flattened outputs
        flattened_concat = torch.cat([layer(combined_output).flatten() for layer in self.pooling_layers], dim=1)

        # Fully connected layers
        x = flattened_concat
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)