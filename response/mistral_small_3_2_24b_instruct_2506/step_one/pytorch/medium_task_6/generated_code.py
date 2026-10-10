import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Three parallel blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initial convolution
        initial_out = self.initial_conv(x)

        # Parallel blocks
        block1_out = self.block1(initial_out)
        block2_out = self.block2(initial_out)
        block3_out = self.block3(initial_out)

        # Add outputs
        added_out = torch.add(initial_out, block1_out)
        added_out = torch.add(added_out, block2_out)
        added_out = torch.add(added_out, block3_out)

        # Flatten and fully connected layers
        flattened = added_out.view(added_out.size(0), -1)
        fc_out = self.fc1(flattened)
        fc_out = self.fc2(fc_out)

        # Softmax output
        output = self.softmax(fc_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print("TensorShape:", output_tensor.shape)