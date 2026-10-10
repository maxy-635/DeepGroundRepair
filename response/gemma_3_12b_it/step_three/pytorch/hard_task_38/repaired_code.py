import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the repeated block structure
        def create_block(in_channels):
            layers = []  # output tensor shape: (unknown)
            layers.append(nn.BatchNorm2d(in_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            return nn.Sequential(*layers)

        # Pathway 1
        self.pathway1 = nn.Sequential(
            create_block(1),  # Input channels for MNIST is 1
            create_block(1),
            create_block(1)
        )

        # Pathway 2
        self.pathway2 = nn.Sequential(
            create_block(1),
            create_block(1),
            create_block(1)
        )

        # Concatenate outputs from both pathways
        self.concat = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Classification using fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 28 * 28, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)  # 10 classes for MNIST
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass through pathway 1
        x1 = self.pathway1(x)  # output tensor shape: (1, 1, 28, 28)

        # Pass through pathway 2
        x2 = self.pathway2(x)  # output tensor shape: (1, 1, 28, 28)

        # Concatenate the outputs of both pathways
        x_concat = torch.cat((x1, x2), dim=1)  # output tensor shape: (1, 2, 28, 28)

        # Apply convolution after concatenation
        x_merged = self.concat(x_concat)  # output tensor shape: (1, 32, 28, 28)

        # Classify using fully connected layers
        x_output = self.classifier(x_merged)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (1 batch, 1 channel, 28x28 image)
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)  # Expected: torch.Size([1, 10])