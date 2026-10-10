import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.main_path_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.main_path_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
        )

        # Fusion and Classification
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8 + 192 * 8 * 8, out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path_1x1_out = self.main_path_1x1(x)  # output tensor shape: (1, 64, 16, 16)
        main_path_3x3_out = self.main_path_3x3(x)  # output tensor shape: (1, 64, 16, 16)
        main_path_5x5_out = self.main_path_5x5(x)  # output tensor shape: (1, 64, 16, 16)

        # Concatenate Main Path Outputs
        main_path_concat = torch.cat([main_path_1x1_out, main_path_3x3_out, main_path_5x5_out], dim=1)  # output tensor shape: (1, 192, 16, 16)

        # Branch Path
        branch_path_out = self.branch_path(x)  # output tensor shape of this layer should be (1, 192, 16, 16)

        # Fusion
        fused_features = main_path_concat + branch_path_out

        # Classification
        flattened_features = self.flatten(fused_features)
        dropped_features = self.dropout(flattened_features)
        fc1_out = self.fc1(dropped_features)
        relu_out = self.relu(fc1_out)
        output = self.fc2(relu_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)