import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: Average Pooling + 1x1 convolution
        path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Path 3: 1x1 convolution + parallel 1x3 and 3x1 convolutions
        path3_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        path3_conv1x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=1)
        path3_conv3x1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=1)
        path3 = nn.Sequential(
            path3_conv1,
            nn.ModuleList([path3_conv1x3, path3_conv3x1]),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Path 4: 1x1 convolution + 3x3 convolution + parallel 1x3 and 3x1 convolutions
        path4_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        path4_conv3x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        path4_conv1x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=1)
        path4_conv3x1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=1)

        path4 = nn.Sequential(
            path4_conv1,
            path4_conv3x3,
            nn.ModuleList([path4_conv1x3, path4_conv3x1]),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate the outputs of all paths
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 32 * 32, out_features=10)
        )

    def forward(self, x):
        # Extract features using each path
        path1_output = path1(x)
        path2_output = path2(x)
        path3_output = path3(x)
        path4_output = path4(x)

        # Concatenate the features
        concatenated_features = torch.cat((path1_output, path2_output, path3_output, path4_output), dim=1)

        # Feature fusion
        fused_features = self.feature_fusion(concatenated_features)

        # Flatten the features for the classifier
        flattened_features = fused_features.view(-1, 64 * 32 * 32)

        # Classify the features
        output = self.classifier(flattened_features)
        probabilities = F.softmax(output, dim=1)

        return probabilities


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])