import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Primary path
        self.primary_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Changed groups=32 to achieve depthwise effect
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, groups=32),  # Changed groups=32 to achieve depthwise effect
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate the outputs of both paths
        self.concat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        )

        # Block 2: Channel Shuffling
        self.channel_shuffle = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.Permute((0, 2, 3, 1)),
            nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layer
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        # Block 1: Primary path
        primary_output = self.primary_path(x)

        # Branch path
        branch_output = self.branch_path(x)

        # Concatenate along channel dimension
        concatenated = torch.cat((primary_output, branch_output), dim=1)

        # Block 2: Channel Shuffling
        features_shape = concatenated.shape
        reshaped = concatenated.view(features_shape[0], features_shape[1], features_shape[2], features_shape[3] // 2, 2)
        reshaped = reshaped.permute(0, 1, 4, 2, 3)
        reshaped = reshaped.view(features_shape[0], features_shape[1], features_shape[2] * features_shape[3] // 2)

        # Apply the channel shuffle block
        shuffled = self.channel_shuffle(reshaped)

        # Flatten and pass through fully connected layer
        flattened = shuffled.view(shuffled.size(0), -1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)