import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path and branch path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Second block: three max pooling layers with different scales
        self.pooling_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        )

        # Flatten and concatenate
        self.flatten = nn.Flatten()
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features='mask', out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block processing
        main_output = self.main_path(x)  # output tensor shape: (1, 1, 28, 28)
        branch_output = x  # output tensor shape: (1, 1, 28, 28)
        block1_output = torch.add(main_output, branch_output)  # output tensor shape: (1, 1, 28, 28)

        # Second block processing
        pool1 = self.pooling_block[0](block1_output)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.pooling_block[1](block1_output)  # output tensor shape: (1, 1, 14, 14)
        pool3 = self.pooling_block[2](block1_output)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        flat_pool1 = self.flatten(pool1)  # output tensor shape: (1, 784)
        flat_pool2 = self.flatten(pool2)  # output tensor shape: (1, 196)
        flat_pool3 = self.flatten(pool3)  # output tensor shape: (1, 49)

        # Concatenate along dim=1
        concat_output = torch.cat((flat_pool1, flat_pool2, flat_pool3), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layers
        fc_output = self.fc(concat_output)

        return fc_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)