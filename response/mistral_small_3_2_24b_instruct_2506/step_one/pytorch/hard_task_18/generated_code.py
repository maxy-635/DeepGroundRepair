import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Second block: Channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_classifier = nn.Linear(in_features=64*8*8, out_features=10)  # CIFAR-10 has 10 classes
        self.flatten = nn.Flatten()

    def forward(self, x):
        # First block
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.avg_pool(out)

        # Second block
        # Main path
        main_path = self.global_avg_pool(out)
        main_path = self.flatten(main_path)
        main_path = self.fc1(main_path)
        main_path = torch.relu(main_path)
        main_path = self.fc2(main_path)
        main_path = torch.sigmoid(main_path)
        main_path = torch.reshape(main_path, (-1, 64, 1, 1))

        # Channel attention
        out = torch.mul(out, main_path)

        # Classification
        out = self.flatten(out)
        out = self.fc_classifier(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])