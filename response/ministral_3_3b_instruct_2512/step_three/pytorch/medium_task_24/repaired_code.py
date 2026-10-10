import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Branch 3: <max pooling>
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Dropout(0.2)
        )

        # Concatenate all branches
        self.concatenate = nn.Sequential(
            nn.Conv2d(32*3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        branch1_out = self.branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 32, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate outputs from all branches
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        concatenated = self.concatenate(concatenated)

        # Final classification
        out = self.fc(concatenated)
        return out

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)

    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)