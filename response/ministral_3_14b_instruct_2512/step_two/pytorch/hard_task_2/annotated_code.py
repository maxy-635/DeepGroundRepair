import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the channel dimension
        self.split_conv = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)

        # Three parallel branches with 1x1 -> 3x3 -> 1x1 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)
        )

        # Main path after combining branches
        self.main_path = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 32 * 32, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Split input into three groups along channel dimension
        x_split = self.split_conv(x).chunk(3, dim=1)

        # Process each branch
        branch1_out = self.branch1(x_split[0])
        branch2_out = self.branch2(x_split[1])
        branch3_out = self.branch3(x_split[2])

        # Combine branches with addition
        combined = branch1_out + branch2_out + branch3_out

        # Add to original input (after splitting)
        original = x.chunk(3, dim=1)
        combined = combined + torch.cat(original, dim=1)

        # Process through main path
        main_path_out = self.main_path(combined)

        # Flatten and classify
        flattened = self.flatten(main_path_out)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)