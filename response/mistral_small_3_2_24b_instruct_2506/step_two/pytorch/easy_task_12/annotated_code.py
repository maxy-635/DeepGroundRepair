import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*14*14, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path output
        main_out = self.main_path(x)  # output tensor shape of this layer should be (1, 32, 28, 28)

        # Branch path output
        branch_out = self.branch_path(x)  # output tensor shape: (1, 32, 28, 28)

        # Sum the outputs
        combined = torch.add(main_out, branch_out)

        # Flatten and pass through fully connected layer
        out = self.flatten(combined)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)