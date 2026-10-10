import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Path 2: 1x1 conv followed by 1x7 and 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 conv followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Concatenate all paths and add final layers
        self.fc = nn.Linear(in_features=64 * 4 * 32 * 32, out_features=10)  # Assuming CIFAR-10 has 10 classes

    def forward(self, x):
        # Process each path
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate along channel dimension
        concat_out = torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)

        # Flatten and pass through fully connected layer
        flattened = torch.flatten(concat_out, 1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)