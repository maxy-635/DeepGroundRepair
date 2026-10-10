import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1_bn = nn.BatchNorm2d(num_features=64)
        self.block1_relu = nn.ReLU(inplace=True)
        self.block2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_bn = nn.BatchNorm2d(num_features=64)
        self.block2_relu = nn.ReLU(inplace=True)
        self.block3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block3_bn = nn.BatchNorm2d(num_features=64)
        self.block3_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        initial_output = self.initial_conv(x)  # output tensor shape: (1, 64, 32, 32)
        
        block1_output = self.block1_conv(x)
        block1_output = self.block1_bn(block1_output)
        block1_output = self.block1_relu(block1_output)
        
        block2_output = self.block2_conv(x)
        block2_output = self.block2_bn(block2_output)
        block2_output = self.block2_relu(block2_output)
        
        block3_output = self.block3_conv(x)
        block3_output = self.block3_bn(block3_output)
        block3_output = self.block3_relu(block3_output)
        
        added_output = torch.cat((initial_output, block1_output, block2_output, block3_output), dim=1)
        
        flattened_output = self.flatten(added_output)
        fc1_output = self.fc1(flattened_output)
        fc2_output = self.fc2(fc1_output)
        output = self.softmax(fc2_output)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)