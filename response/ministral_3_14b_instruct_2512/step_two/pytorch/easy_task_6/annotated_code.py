import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Branch path (direct connection)
        self.branch = nn.Sequential()

        # Combination layer
        self.add = torch.add

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=784, out_features=10)  # MNIST images are 28x28=784 pixels

    def forward(self, x):
        # Main path
        main_path = self.relu(self.conv1(x))
        main_path = self.conv2(main_path)

        # Branch path (identity connection)
        branch_path = x

        # Combine paths
        combined = self.add(main_path, branch_path)

        # Classification
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 1 channel, 28x28 pixels
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10] for 10 classes