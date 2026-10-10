import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.block1_fc1 = nn.Linear(in_features=32, out_features=32)
        self.block1_fc2 = nn.Linear(in_features=32, out_features=32)

        # Block 2
        self.block2_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.block2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fusion and Classification
        self.block3_fc1 = nn.Linear(in_features=128 * 8 * 8 + 32, out_features=128)
        self.block3_fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        block1_output = self.block1_avgpool(x)
        block1_output = torch.reshape(block1_output, (-1, 32))  # Assuming input has 32 channels
        block1_output = F.relu(self.block1_fc1(block1_output))
        block1_output = F.relu(self.block1_fc2(block1_output))
        block1_output = torch.reshape(block1_output, (-1, 32, 1, 1))

        # Block 2
        block2_output = F.relu(self.block2_conv1(x))
        block2_output = F.relu(self.block2_conv2(block2_output))
        block2_output = self.block2_maxpool(block2_output)

        # Fusion
        combined_output = torch.add(block2_output, block1_output)

        # Classification
        combined_output = torch.flatten(combined_output, 1)
        classification_output = F.relu(self.block3_fc1(combined_output))
        classification_output = self.block3_fc2(classification_output)

        return classification_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)