import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)

        # Second block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)

        # Third block
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)

        # Parallel branch
        self.conv_parallel = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Process through sequential blocks
        out1 = self.relu1(self.bn1(self.conv1(x)))
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.relu3(self.bn3(self.conv3(out2)))

        # Process through parallel branch
        out_parallel = self.conv_parallel(x)

        # Add all outputs
        combined = out1 + out2 + out3 + out_parallel

        # Flatten and pass through fully connected layers
        combined = combined.view(combined.size(0), -1)
        out = self.fc2(self.fc1(combined))

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(2, 3, 32, 32)  # Batch size of 2, 3 channels, 32x32 images
    output_tensor = model(input_tensor)
    print(model)
    print(output_tensor)