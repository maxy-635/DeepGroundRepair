import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.pool2(x1)  # output tensor shape: (1, 3, 16, 16)
        x3 = self.pool3(x2)  # output tensor shape: (1, 3, 4, 4)
        x1_flat = self.flatten1(x1)  # output tensor shape: (1, 3072)
        x2_flat = self.flatten2(x2)  # output tensor shape: (1, 768)
        x3_flat = self.flatten3(x3)  # output tensor shape: (1, 48)
        concatenated = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        x = self.fc1(concatenated)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)