import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=1024, out_features=128)
        self.relu_linear = nn.ReLU(inplace=True)
        self.output = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32)
        x = self.conv1(x)  # output tensor shape: (1, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 64, 32)
        x = self.relu2(x)  # output tensor shape: (1, 64, 32)
        x = self.conv3(x)  # output tensor shape: (1, 128, 32)
        x = self.relu3(x)  # output tensor shape: (1, 128, 32)
        x = self.pool(x)  # output tensor shape: (1, 128, 16)
        x = self.conv4(x)  # output tensor shape: (1, 64, 16)
        x = self.relu4(x)  # output tensor shape: (1, 64, 16)
        x = self.flatten(x)  # output tensor shape: (1, 1024)
        x = self.linear(x)
        x = self.relu_linear(x)
        x = self.output(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Batch size 1, 3 channels, 32 features
    output_tensor = model(input_tensor)
    print(output_tensor.shape)