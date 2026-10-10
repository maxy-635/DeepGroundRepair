import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Two convolutional blocks followed by average pooling
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Path 2: Single convolutional layer
        self.path2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)  # Assuming input size after path1 is 8x8
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        path1_output = self.path1(x)  # output tensor shape of this layer should be (1, 64, 16, 16)
        path2_output = self.path2(x)  # output tensor shape: (1, 32, 32, 32)
        combined_output = path1_output + path2_output
        output = self.classifier(combined_output)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 images are 32x32
    output_tensor = DL_Model()(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)