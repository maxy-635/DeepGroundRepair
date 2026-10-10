import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.model = nn.Sequential(
            # 1x1 convolution to reduce dimensionality
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            # 3x3 convolution to extract features
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # 1x1 convolution to restore dimensionality
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            # Flatten the output
            nn.Flatten(),
            # Fully connected layer for classification
            nn.Linear(in_features=32*28*28, out_features=10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)