import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=3),  # Depthwise convolution
            nn.LayerNorm(normalized_shape=[3, 32, 32]),  # Assuming CIFAR-10 image size is 32x32
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),  # 1x1 pointwise conv
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),  # 1x1 pointwise conv
            nn.ReLU(inplace=True)
        )

        # Branch path (identity connection)
        self.branch_path = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*32*32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        # Main path output
        main_out = self.main_path(x)

        # Branch path output (identity)
        branch_out = self.branch_path(x)

        # Combine outputs
        combined = torch.add(main_out, branch_out)

        # Classification
        output = self.classifier(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)