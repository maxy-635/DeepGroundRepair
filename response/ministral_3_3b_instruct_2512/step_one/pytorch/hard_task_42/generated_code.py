import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import ReshapeTransform

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel max pooling paths with different scales
        self.pooling_paths = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(kernel_size=1, stride=1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
                nn.Flatten()
            )
        ])

        # Dropout layers for regularization
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(p=0.5),
            nn.Dropout(p=0.5),
            nn.Dropout(p=0.5)
        ])

        # Concatenate the flattened outputs
        self.concat_layer = nn.Sequential(
            nn.Linear(28 * 28 * 3, 1024),  # Adjust input features based on pooling results
            ReshapeTransform(in_shape=(1024, 1, 1))
        )

        # Block 2: Parallel convolutional paths
        self.conv_paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=7, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=7, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=7, stride=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(1, 32, kernel_size=1, stride=1),
                nn.ReLU()
            )
        ])

        # Concatenate the outputs of all paths
        self.concat_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 1 * 1, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Block 1 processing
        pooled_outputs = []
        for i, path in enumerate(self.pooling_paths):
            pooled_output = path(x)
            pooled_output = self.dropout_layers[i](pooled_output)
            pooled_outputs.append(pooled_output)

        # Concatenate the outputs
        concatenated = torch.cat(pooled_outputs, dim=1)
        flattened = self.concat_layer(concatenated)

        # Block 2 processing
        conv_outputs = []
        for path in self.conv_paths:
            conv_output = path(flattened)
            conv_outputs.append(conv_output)

        # Concatenate the outputs of all paths
        concatenated_conv = torch.cat(conv_outputs, dim=1)
        concatenated_conv = self.concat_conv(concatenated_conv)

        # Final classification
        output = self.fc_layers(concatenated_conv.view(-1, 64 * 1 * 1))
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)