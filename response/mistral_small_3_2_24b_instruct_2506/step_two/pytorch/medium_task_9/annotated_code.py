import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # First basic block
        self.block1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
            )
        )

        # Second basic block
        self.block2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
            )
        )

        # Average pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Fully connected layer
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # First block
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.block1[0](x)  # output tensor shape: (1, 16, 32, 32)
        x = x + residual  # output tensor shape: (1, 16, 32, 32)

        # Second block
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.block2[0](x)  # output tensor shape: (1, 16, 32, 32)
        x = x + residual  # output tensor shape: (1, 16, 32, 32)

        # Average pooling and flatten
        x = self.avgpool(x)  # output tensor shape: (1, 16, 29, 29)
        x = self.flatten(x)  # output tensor shape: (1, 13456)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)