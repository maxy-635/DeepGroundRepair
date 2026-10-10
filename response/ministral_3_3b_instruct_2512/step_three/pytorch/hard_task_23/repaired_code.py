import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Local feature extraction branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 1: Downsampling and Upsampling
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        )

        # Branch 2: Downsampling and Upsampling
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        )

        # Concatenation and final convolution
        self.concatenate = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.classifier = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        local_features = self.local_branch(x)  # output tensor shape: (1, 64, 32, 32)
        branch1_features = self.branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch2_features = self.branch2(x)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate features from all branches
        concatenated_features = torch.cat((local_features, branch1_features, branch2_features), dim=1)  # output tensor shape: (1, 128, 32, 32)
        concatenated_features = self.concatenate(concatenated_features)

        # Flatten and pass through fully connected layer
        flattened_features = concatenated_features.view(concatenated_features.size(0), -1)
        output = self.classifier(flattened_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor)