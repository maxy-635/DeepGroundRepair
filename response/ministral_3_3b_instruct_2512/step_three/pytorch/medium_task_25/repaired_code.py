import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True)
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the four paths
        self.concatenate = nn.Sequential(
            nn.Conv2d(64 * 4, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the outputs of the four paths
        concatenated = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        concatenated = self.concatenate(concatenated)

        # Classification
        output = self.fc(concatenated)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)