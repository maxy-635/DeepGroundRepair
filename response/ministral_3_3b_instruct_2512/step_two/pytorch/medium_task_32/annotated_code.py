import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the last dimension
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, depthwise=True, groups=3),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Flatten and classify
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 32 * 32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)

        # Concatenate along the channel dimension
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_concat = self.concat(x_concat)

        x_flatten = self.flatten(x_concat)
        output = self.fc(x_flatten)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)