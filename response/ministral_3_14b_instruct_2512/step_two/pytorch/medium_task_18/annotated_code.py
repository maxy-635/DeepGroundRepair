import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction layers with different kernel sizes
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Concatenate the features
        self.concat = nn.ModuleList([
            self.conv1x1,
            self.conv3x3,
            self.conv5x5
        ])

        # Additional convolutional layers after concatenation
        self.conv_after_concat = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Extract features at multiple scales
        conv1x1_out = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the features along the channel dimension
        concat_out = torch.cat([conv1x1_out, conv3x3_out, conv5x5_out], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Additional processing
        x = self.conv_after_concat(concat_out)  # output tensor shape: (1, 128, 16, 16)

        # Classification
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)