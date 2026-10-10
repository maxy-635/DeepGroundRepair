import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        )
        
        # Classification
        self.classifier = nn.Linear(in_features=128 * 8 * 8, out_features=10) # Assuming input size after fusion is 8x8

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Extract features from each path
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 16, 16)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Reshape path3 and path4 outputs to be compatible for concatenation
        path3_out = path3_out.view(path3_out.size(0), -1)
        path4_out = path4_out.view(path4_out.size(0), -1)

        # Concatenate the outputs of all paths
        combined_features = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)

        # Feature fusion
        fused_features = self.fusion(combined_features)

        # Flatten the features for classification
        flattened_features = fused_features.view(fused_features.size(0), -1)

        # Classification
        output = self.classifier(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)