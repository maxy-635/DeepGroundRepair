import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First branch: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Second branch: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Third branch: max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each branch
        out1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        out3 = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3), dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Flatten and pass through fully connected layers
        out = self.flatten(out)  # output tensor shape: (1, 196608)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print("TensorShape:", output_tensor.shape)
    print("TensorValue:", output_tensor)