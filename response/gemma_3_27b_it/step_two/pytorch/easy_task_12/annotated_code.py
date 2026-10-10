import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_block1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.main_path_block2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch Path
        self.branch_path_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size='mask', stride='mask', padding='mask')

        # Flatten and Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main Path
        main_path_output = self.main_path_block1(x)  # output tensor shape: (1, 16, 14, 14)
        main_path_output = self.main_path_block2(main_path_output)  # output tensor shape: (1, 32, 7, 7)

        # Branch Path
        branch_path_output = self.branch_path_conv(x)  # output tensor shape of this layer should be (1, 32, 7, 7)

        # Sum the outputs
        combined_output = main_path_output + branch_path_output

        # Flatten and Fully Connected Layer
        flattened_output = self.flatten(combined_output)
        output = self.fc(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)