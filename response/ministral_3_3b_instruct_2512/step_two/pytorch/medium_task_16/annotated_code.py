import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split input into three groups along the channel dimension
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        # Apply average pooling to each group
        self.pool = nn.AdaptiveAvgPool2d(output_size=(None, 1))

        # Concatenate the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(output_size=(None, 1))
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split and process each group
        x1 = self.conv1[0](x)  # output tensor shape: (1, 1, 32, 32)
        x2 = self.conv1[1](x1)  # output tensor shape: (1, 1, 32, 32)
        x3 = self.conv1[2](x2)  # output tensor shape: (1, 1, 32, 32)

        # Apply pooling to each group
        x1_pool = self.pool(x1)  # output tensor shape: (1, 1, 32, 1)
        x2_pool = self.pool(x2)  # output tensor shape: (1, 1, 32, 1)
        x3_pool = self.pool(x3)  # output tensor shape: (1, 1, 32, 1)

        # Concatenate the pooled feature maps
        x_concat = torch.cat((x1_pool, x2_pool, x3_pool), dim=1)  # output tensor shape: (1, 3, 32, 1)

        # Flatten and pass through fully connected layers
        x_flatten = self.flatten(x_concat)  # output tensor shape: (1, 96)
        x_fc = self.fc1(x_flatten)

        return x_fc

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input shape for CIFAR-10 images
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)