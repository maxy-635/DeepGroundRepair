import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Path 2: 1x1 + two 3x3 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 + single 3x3 convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Path 4: max pooling + 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenate paths and flatten
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=64*32*32, out_features=128)  # Assuming input size 32x32 after pooling
        self.output = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Get outputs from each path
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate along channel dimension
        concat_out = torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)

        # Flatten and pass through dense layers
        flattened = self.flatten(concat_out)
        dense_out = self.dense(flattened)
        output = self.output(dense_out)
        output = self.softmax(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)