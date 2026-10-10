import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Branch path
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Main path
        main_path = x
        main_path = self.conv1(main_path)
        main_path = self.pool1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.pool2(main_path)
        main_path = self.conv3(main_path)
        main_path = self.pool3(main_path)
        main_path = main_path.view(-1, 64)  # Flatten for FC layers

        # Branch path
        branch_path = self.avgpool(x)
        branch_path = torch.flatten(branch_path)
        branch_path = self.fc1(branch_path)
        branch_path = self.fc2(branch_path)

        # Add the two paths
        output = torch.cat((main_path, branch_path), dim=1)

        # Final classification
        output = nn.Sigmoid()(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)