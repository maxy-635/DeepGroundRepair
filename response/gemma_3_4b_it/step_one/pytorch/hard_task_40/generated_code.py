import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc1 = nn.Linear(1 * 1 * 1, 128)
        self.reshape = nn.Reshape((128, 1, 1, 1))

        # Second block
        self.path1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Concatenation
        self.concat = nn.Concat([self.path1, self.path2, self.path3, self.path4])

        # Output layers
        self.fc2 = nn.Linear(32 * 4, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First block
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        x = self.path1(x)
        x = self.path2(x)
        x = self.path3(x)
        x = self.path4(x)
        x = self.concat(x)

        # Output layers
        x = self.fc2(x)
        x = self.relu(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)