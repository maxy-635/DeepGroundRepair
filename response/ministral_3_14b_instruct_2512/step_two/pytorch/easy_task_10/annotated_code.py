import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.features = nn.Sequential(
            # 1x1 convolution to increase dimensionality with stride=2
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),

            # 3x3 depthwise separable convolution for feature extraction with stride=2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),

            # 1x1 convolution to reduce dimensionality with stride=2
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.features(x)  # output tensor shape: (1, 16, 4, 4)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)