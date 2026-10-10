import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.primary_path = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.DepthwiseConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate along channel dimension
        self.concat_layer = nn.Concatenation()

        # Block 2
        self.reshape_layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.permute_layer = nn.Permute(dims=(3, 2, 1))
        self.reshape_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.UnAdaptiveAvgPool2d((4, 4))
        )


        # Fully connected layer
        self.fc_layer = nn.Linear(64, 10)

    def forward(self, x):
        # Block 1
        primary_features = self.primary_path(x)

        # Branch path
        branch_features = self.branch_path(x)

        # Concatenate
        concatenated_features = self.concat_layer(primary_features, branch_features)

        # Block 2
        reshaped_features = self.reshape_layer(concatenated_features)
        permuted_features = self.permute_layer(reshaped_features)
        reshaped_features2 = self.reshape_layer2(permuted_features)

        # Fully connected layer
        output = self.fc_layer(reshaped_features2)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape: batch_size, channels, height, width
    output_tensor = model(input_tensor)
    print(output_tensor.shape)