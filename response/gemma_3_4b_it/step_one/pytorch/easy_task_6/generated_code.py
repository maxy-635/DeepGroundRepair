import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.branch_conv = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(10*28*28, 10)

    def forward(self, x):
        main_path = self.conv1(x)
        main_path = self.relu1(main_path)
        main_path = self.conv2(main_path)
        
        branch_path = self.branch_conv(x)
        
        combined = torch.add(main_path, branch_path)
        
        flattened = self.flatten(combined)
        output = self.linear(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)