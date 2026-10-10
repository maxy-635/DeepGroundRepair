import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.feature_extraction_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.feature_extraction_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=401408, out_features=1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=1000)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 224, 224)
        x = self.feature_extraction_1(x)  # output tensor shape: (1, 32, 112, 112)
        x = self.feature_extraction_2(x)  # output tensor shape: (1, 64, 56, 56)
        x = self.conv_layers(x)  # output tensor shape: (1, 512, 56, 56)
        x = self.avg_pool(x)  # output tensor shape: (1, 512, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 401408)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 224, 224)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)