import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three groups of convolutions
        self.conv1x1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        self.conv1x1_3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.conv1x1_5 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1x1_6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Define the fully connected layer
        self.fc = nn.Linear(in_features=32*32*32, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        x1, x2, x3 = torch.split(x, split_size_or_sections=1, dim=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Process each group through the convolutional layers
        out1 = self.conv1x1_2(self.conv3x3_1(self.conv1x1_1(x1)))
        out2 = self.conv1x1_4(self.conv3x3_3(self.conv1x1_3(x2)))
        out3 = self.conv1x1_6(self.conv3x3_5(self.conv1x1_5(x3)))

        # Combine the outputs from the three groups
        combined = torch.add(out1, out2)
        combined = torch.add(combined, out3)

        # Fuse with the original input
        fused = torch.add(combined, x)

        # Flatten and pass through the fully connected layer
        flattened = self.flatten(fused)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)