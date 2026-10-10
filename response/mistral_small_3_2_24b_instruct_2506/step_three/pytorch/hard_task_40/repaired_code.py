import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average pooling layers and concatenation
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1029, out_features=128)
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=4*4*4*4)
        )

        # Second block: Multi-scale feature extraction paths
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=16*4*4*4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for MNIST

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x1 = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x3 = self.avg_pool3(x)  # output tensor shape: (1, 1, 7, 7)

        x1_flat = self.flatten(x1)  # output tensor shape: (1, 784)
        x2_flat = self.flatten(x2)  # output tensor shape: (1, 196)
        x3_flat = self.flatten(x3)  # output tensor shape: (1, 49)

        x_concat = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)  # output tensor shape: (1, 1029)
        x_fc = self.fc1(x_concat)
        x_reshape = self.reshape(x_fc)

        # Second block
        path1_out = self.path1(x_reshape)
        path2_out = self.path2(x_reshape)
        path3_out = self.path3(x_reshape)
        path4_out = self.path4(x_reshape)

        x_concat_paths = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        x_flat = self.flatten(x_concat_paths)

        # Final fully connected layers
        x_fc2 = self.fc2(x_flat)
        x_fc3 = self.fc3(x_fc2)

        return x_fc3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)