import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Path 2: 1x1 conv + two 3x3 convs
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 conv + 3x3 conv
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 4: Max pooling + 1x1 conv
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenate paths
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass through parallel paths
        path1_out = self.path1(x)  # output tensor shape: (1, 32, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 32, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 32, 32, 32)
        path4_out = self.path4(x)  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Concatenate outputs
        combined = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)

        # Flatten and pass through dense layers
        output = self.concat(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10: 32x32 images with 3 channels)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)