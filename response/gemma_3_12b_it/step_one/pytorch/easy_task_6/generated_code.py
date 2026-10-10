import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=784, out_features=10)

        # Combination
        self.addition = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1*28*28, out_features=784),
        )


    def forward(self, x):
        # Main path
        main_path = self.conv1(x)
        main_path = self.relu1(main_path)
        main_path = self.conv2(main_path)

        # Branch path
        branch_path = x

        # Combine paths
        combined = torch.add(main_path, branch_path)

        # Flatten and classify
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)