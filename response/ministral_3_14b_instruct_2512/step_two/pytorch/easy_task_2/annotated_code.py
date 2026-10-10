import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            # First feature extraction block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # Second feature extraction block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # Additional convolutional layers
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            # Flatten layer
            nn.Flatten()
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 28 * 28, out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1000),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)