import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Max Pooling Layers
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        )

        # Flatten and Concatenate
        self.flatten1 = nn.Flatten()
        self.cat1 = nn.Sequential(
            nn.Flatten(),
            # Dummy layers to enable concatenation
            nn.Flatten(),
            nn.Flatten()
        )

        # Fully Connected Layer and Reshape
        self.fc1 = nn.Linear(in_features=1, out_features=1)
        self.reshape1 = nn.Reshape(shape=(-1,))

        # Block 2: Convolutional and Max Pooling Branches
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.pool3x3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # Concatenate Block 2 Outputs
        self.cat2 = nn.Sequential(
            nn.Flatten(),
            nn.Flatten(),
            nn.Flatten(),
            nn.Flatten()
        )
        
        # Classification
        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=1, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.block1(x)
        x = self.flatten1(x)
        x = self.cat1(x)

        # Fully Connected Layer and Reshape
        x = self.fc1(x)
        x = self.reshape1(x)

        # Block 2
        x = torch.cat((self.conv1x1(x), self.conv3x3(x), self.conv5x5(x), self.pool3x3(x)), dim=1)
        x = self.cat2(x)

        # Classification
        x = self.flatten2(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input: batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)