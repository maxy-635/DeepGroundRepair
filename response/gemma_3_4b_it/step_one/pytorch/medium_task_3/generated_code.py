import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Adjust the input size based on the convolutional output

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat((x, x), dim=1)  # Concatenate the outputs from both paths
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input: batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)