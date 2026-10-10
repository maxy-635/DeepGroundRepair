import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Three max pooling layers with different window sizes
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply max pooling and flatten
        x1 = self.flatten1(self.maxpool1(x))  # output tensor shape: (1, 3072)
        x2 = self.flatten2(self.maxpool2(x))  # output tensor shape: (1, 768)
        x3 = self.flatten3(self.maxpool3(x))  # output tensor shape: (1, 192)

        # Concatenate the flattened outputs
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 4032)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])