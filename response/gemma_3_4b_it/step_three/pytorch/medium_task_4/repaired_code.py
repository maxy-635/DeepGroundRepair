import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            F.relu(torch.nn.functional.conv2d(input, weight, bias=None)),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            F.relu(torch.nn.functional.conv2d(input, weight, bias=None)),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            F.relu(torch.nn.functional.conv2d(input, weight, bias=None))
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 32, 10)  # Adjust output size based on input image size

    def forward(self, x):
        path1_output = self.path1(x)
        path2_output = self.path2(x)
        merged = torch.cat((path1_output.view(-1, 32 * 32 * 32), path2_output.view(-1, 32 * 32 * 32)), dim=1)
        output = self.linear(merged)
        return F.softmax(output, dim=1)


if __name__ == "__main__":
    # Define input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    # Assuming 'weight' is a global variable or defined elsewhere
    weight = torch.randn(16, 3, 3, 3)  # Example weight tensor
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)