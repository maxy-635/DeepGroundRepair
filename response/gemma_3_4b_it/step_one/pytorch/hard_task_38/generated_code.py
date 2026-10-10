import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the first processing pathway
        self.conv_block1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Define the second processing pathway
        self.conv_block2 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Merge the outputs from both pathways
        self.merge = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Classification layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes (MNIST)

    def forward(self, x):
        # Forward pass through the first pathway
        path1 = self.conv_block1(x)

        # Forward pass through the second pathway
        path2 = self.conv_block2(x)

        # Merge the pathways
        merged = self.merge(torch.cat((path1, path2), dim=1))

        # Classification
        x = merged.view(-1, -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Define the input shape
    input_shape = (1, 28, 28, 1)  # Batch size, height, width, channels

    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = torch.randn(1, *input_shape)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)