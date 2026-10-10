import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Four parallel convolutional paths
        path1 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        path2 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        path3 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        path4 = nn.Sequential(  # output tensor shape: (unknown)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.concat = nn.Sequential(
            path1,
            path2,
            path3,
            path4
        )
        
        # Flatten and dense layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass input through each path
        out1 = self.concat[0](x)  # output tensor shape: (1, 32, 32, 32)
        out2 = self.concat[1](x)  # output tensor shape: (1, 32, 32, 32)
        out3 = self.concat[2](x)  # output tensor shape: (1, 32, 32, 32)
        out4 = self.concat[3](x)  # output tensor shape: (1, 32, 16, 16)
        

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3, out4), dim=1)

        # Flatten and pass through dense layer
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)