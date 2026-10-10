import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Step 1: Compress input channels with a 1x1 convolutional layer
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Step 2: Expand features through two parallel convolutional layers
        self.conv1x1_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Step 3: Concatenate the results of the parallel layers
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Step 4: Flatten the output feature map and pass through fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # Step 1: Compress input channels
        x = self.conv1x1(x)

        # Step 2: Expand features through parallel convolutional layers
        conv1x1_output = self.conv1x1_parallel(x)
        conv3x3_output = self.conv3x3_parallel(x)

        # Step 3: Concatenate the outputs
        x = torch.cat((conv1x1_output, conv3x3_output), dim=1)

        # Apply ReLU activation after concatenation
        x = self.concat(x)

        # Step 4: Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (32, 32, 64)
    input_tensor = torch.randn(1, 64, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)