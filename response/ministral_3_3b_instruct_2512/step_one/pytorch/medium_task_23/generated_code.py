import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        # Path 4: average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate the outputs of the four paths
        self.concatenate = nn.Sequential(
            nn.Conv2d(64 * 4, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate the outputs of the four paths
        concatenated = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        concatenated = self.concatenate(concatenated)

        # Flatten and pass through the fully connected layer
        flattened = concatenated.view(concatenated.size(0), -1)
        output = self.classifier(flattened)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 images are 32x32
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)