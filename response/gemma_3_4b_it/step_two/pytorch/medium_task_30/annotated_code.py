import torch
import torch.nn as nn
import torch.utils.data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Average Pooling Layer
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second Average Pooling Layer
        self.pool2 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Third Average Pooling Layer
        self.pool3 = nn.AvgPool2d(kernel_size=1, stride=1)

        # Flatten the outputs of the pooling layers
        self.flatten = nn.Flatten()

        # Concatenate the flattened features
        self.concat = nn.Concatenation()

        # First Fully Connected Layer
        self.fc1 = nn.Linear(in_features=1 * 1 * 1, out_features=128)

        # Second Fully Connected Layer
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Apply Average Pooling
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)

        # Flatten the pooled features
        x = self.flatten(x)

        # Concatenate the flattened features
        x = self.concat([x])

        # Flatten again for the fully connected layers
        x = x.view(-1, 128)

        # Pass through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Define the input tensor shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example: batch size 1, 3 channels (RGB), 32x32 image
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)