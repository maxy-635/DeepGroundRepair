import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=3)  # Assuming CIFAR-10 has 10 classes, adjust output accordingly
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = self.relu1(self.conv1(x))  # output tensor shape: (1, 64, 32, 32)
        main_path = self.adaptive_avg_pool(main_path)  # output tensor shape: (1, 64, 1, 1)
        main_path = self.flatten(main_path)  # output tensor shape: (1, 64)
        main_path = self.fc1(main_path)  # output tensor shape: (1, 128)
        main_path = self.fc2(main_path)  # output tensor shape: (1, 3)
        main_path = main_path.reshape(main_path.shape[0], main_path.shape[1], 1, 1)  # output tensor shape: (1, 3, 1, 1)
        main_path = main_path.expand_as(x)  # output tensor shape: (1, 3, 32, 32)

        # Branch path
        branch_path = x  # output tensor shape: (1, 3, 32, 32)

        # Combine paths
        combined_path = torch.add(main_path, branch_path)  # output tensor shape: (1, 3, 32, 32)
        output = self.fc3(combined_path.flatten(1))

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)