import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1))

        # Branch 1: Local feature extraction with two 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        # Branch 2: Average pooling followed by 3x3 convolution and transposed convolution
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )

        # Branch 3: Same as branch 2
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )

        # Concatenation and 1x1 convolution
        self.concat_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=(1, 1), stride=(1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=6400, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Process through branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate outputs
        concat_out = torch.concat([branch1_out, branch2_out, branch3_out], dim=1)

        # Final 1x1 convolution
        x = self.concat_conv(concat_out)

        # Flatten and fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(model)