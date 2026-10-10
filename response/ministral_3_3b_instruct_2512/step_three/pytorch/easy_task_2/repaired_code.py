import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Two sequential feature extraction layers, each consisting of a convolutional layer followed by average pooling
        self.feature_extraction_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Three additional convolutional layers and another average pooling layer
        self.additional_layers = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # Flatten and fully connected layers with dropout
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=401408, out_features=1024),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=1024),
            nn.Dropout(p=0.5)
        )

        # Softmax output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 224, 224)
        x = self.feature_extraction_layers(x)  # output tensor shape: (1, 64, 56, 56)
        x = self.additional_layers(x)  # output tensor shape: (1, 512, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 401408)
        x = self.fc_layers(x)
        return self.softmax(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 224, 224)  # Dummy input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (batch_size, 1000)