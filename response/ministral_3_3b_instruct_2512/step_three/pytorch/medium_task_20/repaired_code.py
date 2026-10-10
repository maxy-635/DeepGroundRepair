import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Path 2: Two 3x3 convolutions stacked after a 1x1 convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 3: Single 3x3 convolution after a 1x1 convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 4: Max pooling followed by a 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the four paths
        self.concatenate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),  # Assuming input size is 32x32, after pooling it's 16x16
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs of the four paths
        concatenated = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        output = self.concatenate(concatenated)
        return self.output(output)

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 images
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)