import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch path for channel attention
        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=256),
            nn.Sigmoid()
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features='mask', out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path processing
        main_features = self.main_path(x)  # output tensor shape: (1, 256, 16, 16)

        # Get spatial dimensions after main path
        batch_size = main_features.size(0)  # output tensor shape: (1)
        channels = main_features.size(1)  # output tensor shape: (1)
        height = main_features.size(2)  # output tensor shape: (1)
        width = main_features.size(3)  # output tensor shape: (1)

        # Branch path processing
        branch_features = self.branch_path(main_features)  # output tensor shape: (1, 256)
        branch_features = branch_features.view(batch_size, channels, 1, 1)  # output tensor shape: (1, 256, 1, 1)

        # Apply channel attention
        attended_features = main_features * branch_features  # output tensor shape: (1, 256, 16, 16)

        # Flatten for classification
        attended_features = attended_features.view(batch_size, -1)  # output tensor shape: (1, 65536)

        # Final classification
        output = self.classifier(attended_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)