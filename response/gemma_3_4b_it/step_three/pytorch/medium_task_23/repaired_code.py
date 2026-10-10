import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.path1 = nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0)
        # Path 2: 1x1, 1x7, 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding=3)
        )
        # Path 3: 1x1, two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding=3),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(16, 16, kernel_size=7, stride=1, padding=3)
        )
        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=1, padding=2),
            nn.Conv1d(16, 16, kernel_size=1, stride=1, padding=0)
        )
        # Concatenation
        self.concat = nn.Concat(dim=1)
        # Flatten and Linear
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16 * 4, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)
        
        return self.linear(self.flatten(torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)))

if __name__ == "__main__":
    # Dummy input tensor for validation
    batch_size = 4
    channel = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channel, height, width)
    
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)