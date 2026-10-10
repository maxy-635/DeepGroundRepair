import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(1 * 1 * 1 * 3, 128)
        self.reshape = nn.Linear(128, 4 * 128)

        # Second block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        # First block
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.reshape(x)

        # Second block
        x = torch.split(x, 1)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]

        x1 = self.conv1(x1)
        x1 = nn.ReLU(inplace=True)(x1)

        x2 = self.conv2(x2)
        x2 = nn.ReLU(inplace=True)(x2)

        x3 = self.conv3(x3)
        x3 = nn.ReLU(inplace=True)(x3)

        x4 = self.conv4(x4)
        x4 = nn.ReLU(inplace=True)(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Final layer
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)

        # Output layer
        x = self.fc1(x)
        x = nn.Sigmoid()(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)