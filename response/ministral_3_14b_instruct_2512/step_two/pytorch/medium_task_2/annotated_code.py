import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Combined path
        self.combined = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*8*8, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through both paths
        main_features = self.main_path(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        branch_features = self.branch_path(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate features along channel dimension
        combined_features = torch.cat((main_features, branch_features), dim=1)

        # Process through combined path
        output = self.combined(combined_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)