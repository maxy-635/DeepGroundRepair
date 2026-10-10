import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Feature extraction
        self.first_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.first_relu1 = nn.ReLU(inplace=True)
        self.first_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.first_relu2 = nn.ReLU(inplace=True)
        self.first_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Second block: Channel weight generation and application
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming 32x32 input size

    def forward(self, x):
        # First block
        x = self.first_conv1(x)
        x = self.first_relu1(x)
        x = self.first_conv2(x)
        x = self.first_relu2(x)
        x = self.first_avgpool(x)
        x_first_block = x

        # Main path: Global average pooling and channel weight generation
        x_main = self.global_avgpool(x)
        x_main = self.flatten(x_main)
        x_main = self.fc1(x_main)
        x_main = self.relu1(x_main)
        x_main = self.fc2(x_main)
        x_main = self.sigmoid(x_main)
        x_main = x_main.view(x_first_block.shape[0], x_first_block.shape[1], 1, 1)

        # Channel-wise multiplication
        x = x_first_block * x_main

        # Classification
        x = self.flatten(x)
        x = self.fc_classifier(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)