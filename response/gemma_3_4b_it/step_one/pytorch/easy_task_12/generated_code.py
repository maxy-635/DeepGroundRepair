import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 8, 10)  # Adjusted output size

    def forward(self, x):
        main = self.main_path(x)
        branch = self.branch_path(x)

        main = main.view(main.size(0), -1)
        branch = branch.view(branch.size(0), -1)

        x = torch.cat((main, branch), dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        x = nn.Softmax(dim=1)(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)