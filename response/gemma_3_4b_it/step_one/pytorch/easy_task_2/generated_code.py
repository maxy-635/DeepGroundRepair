import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First sequential feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Second sequential feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Additional convolutional layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Another average pooling layer
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjust input size based on pooling
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1000)  # 1000 neurons for softmax

        # Softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224
    output_tensor = model(input_tensor)
    print(output_tensor.shape)