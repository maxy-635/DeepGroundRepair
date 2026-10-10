import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Branch 1: Local feature extraction
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)

        # Branch 2: Downsampling and upsampling
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        # Branch 3: Downsampling and upsampling
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv4_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        # Concatenation
        self.concat = nn.Concatenation()

        # Refinement
        self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.relu6 = nn.ReLU(inplace=True)

        # Fully connected layer
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)

        # Branch 1
        x1 = self.conv2_1(x)
        x1 = self.relu2_1(x1)
        x1 = self.conv3_1(x1)
        x1 = self.relu3_1(x1)

        # Branch 2
        x2 = self.avgpool2(x)
        x2 = self.conv4_2(x2)
        x2 = self.relu4_2(x2)
        x2 = self.conv5_2(x2)
        x2 = self.relu5_2(x2)

        # Branch 3
        x3 = self.avgpool3(x)
        x3 = self.conv4_3(x3)
        x3 = self.relu4_3(x3)
        x3 = self.conv5_3(x3)
        x3 = self.relu5_3(x3)

        # Concatenation
        x = self.concat([x1, x2, x3])

        # Refinement
        x = self.conv6(x)
        x = self.relu6(x)

        # Fully connected layer
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)