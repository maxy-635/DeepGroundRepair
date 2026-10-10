import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Three average pooling layers with different kernel sizes and strides
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3*3072, out_features=512)  # 3 channels * 32*32 / (1*1 + 2*2 + 4*4) = 3072
        self.fc2 = nn.Linear(in_features=512, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Apply three average pooling layers
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool3(x)

        # Flatten each pooled output
        x1_flat = self.flatten1(x1)
        x2_flat = self.flatten1(x2)
        x3_flat = self.flatten1(x3)

        # Concatenate the flattened outputs
        x_concat = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)

        # Further flatten and process through fully connected layers
        x_flat = self.flatten2(x_concat)
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor)