import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the channel dimension
        # and apply 1x1 convolutions to each group independently
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=1)
        )

        # Downsampling via average pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Concatenate the three groups along the channel dimension
        self.concat = nn.Sequential()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Split the input into three groups along the channel dimension
        # and apply 1x1 convolutions to each group independently
        x1 = self.conv1(x[:, :1, :, :])
        x2 = self.conv2(x[:, 1:2, :, :])
        x3 = self.conv3(x[:, 2:, :, :])

        # Downsampling via average pooling
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        # Concatenate the three groups along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)

        # Flatten the concatenated feature maps
        x = self.flatten(x)

        # Fully connected layers for classification
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)