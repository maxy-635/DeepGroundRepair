import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)

        # Global Average Pooling and Fully Connected Layers
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Weighted Feature Map Generation
        self.reshape = nn.Reshape(shape=(64,))  # Reshape for multiplication
        
        # Dimensionality Reduction and Downsampling
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Classification Layer
        self.fc3 = nn.Linear(in_features=32 * 8 * 8, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Initial Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global Average Pooling and Fully Connected Layers
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        # Weighted Feature Map Generation
        x_reshaped = self.reshape(x)
        x_weighted = torch.mul(x_reshaped, x_reshaped)  # Multiply with itself
        x_weighted = x_weighted.reshape(64,1,1)
        x = x_weighted
        x = torch.cat([x,x],dim=0)
        

        # Dimensionality Reduction and Downsampling
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        
        # Classification Layer
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)