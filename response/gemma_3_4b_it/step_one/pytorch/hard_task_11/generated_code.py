import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(512, 10)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        main_path = self.conv1_1(x)
        branch_path = self.conv1_2(x)
        branch_path = self.relu1(branch_path)
        branch_path = self.conv1_3(branch_path)
        branch_path = self.relu1(branch_path)
        main_path = self.conv2_1(main_path)
        main_path = self.relu2(main_path)
        main_path = self.conv2_2(main_path)
        main_path = self.relu2(main_path)
        main_path = self.conv2_3(main_path)

        x = torch.cat((main_path, branch_path), dim=1)
        x = self.conv2_3(x)

        x = torch.add(x, self.conv1_1(x))

        x = self.linear1(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)