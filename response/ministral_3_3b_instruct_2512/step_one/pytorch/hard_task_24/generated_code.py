import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 1: 3x3 convolutional layer for local feature extraction
        self.branch1_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 2: Max pooling, 3x3 convolution, and upsampling
        self.branch2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch2_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            self.branch2_pool,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: Max pooling, 3x3 convolution, and upsampling
        self.branch3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.branch3_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            self.branch3_pool,
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        )

        # Concatenate outputs of all branches and pass through 1x1 convolution
        self.concat_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Branch 1
        branch1_output = self.initial_conv(x)
        branch1_output = self.branch1_conv(branch1_output)

        # Branch 2
        branch2_input = self.initial_conv(x)
        branch2_output = self.branch2_conv(branch2_input)

        # Branch 3
        branch3_input = self.initial_conv(x)
        branch3_output = self.branch3_conv(branch3_input)

        # Concatenate outputs
        concatenated = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        concatenated = self.concat_conv(concatenated)

        # Fully connected layers
        output = self.fc_layers(concatenated)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)