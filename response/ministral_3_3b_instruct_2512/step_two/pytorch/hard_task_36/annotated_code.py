import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=None),  # 3x3 conv for spatial features
            nn.Conv2d(32, 32, kernel_size=1, padding=None),  # 1x1 conv for inter-channel info
            nn.Conv2d(32, 32, kernel_size=1, padding=None),  # Another 1x1 conv
            nn.MaxPool2d(kernel_size=2, stride=2, padding=None),  # Max pooling
            nn.Dropout(0.5)  # 50% dropout
        )

        # Branch pathway
        self.branch_pathway = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=None)  # Convolutional layer matching main pathway output
        )

        # Fusion and final layers
        self.fusion = nn.Sequential(
            nn.Add()  # Addition operation
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 10)  # Fully connected layer for classification

    def forward(self, x):
        main_output = self.main_pathway(x)
        branch_output = self.branch_pathway(x)

        # Fusion
        fused_output = self.fusion(main_output, branch_output)

        # Final layers
        pooled_output = self.avg_pool(fused_output)
        flattened_output = self.flatten(pooled_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # For MNIST, we typically use 1 channel, 28x28 images
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28

    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)