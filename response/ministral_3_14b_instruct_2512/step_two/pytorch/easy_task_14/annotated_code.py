import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Global average pooling to compress spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # First fully connected layer to generate weights
        self.fc1 = nn.Linear(in_features='mask', out_features=32 * 32 * 3)

        # Second fully connected layer to generate weights
        self.fc2 = nn.Linear(in_features=32*32*3, out_features=32*32*3)

        # Sigmoid activation for weights
        self.sigmoid = nn.Sigmoid()

        # Flatten layer for final processing
        self.flatten = nn.Flatten()

        # Final fully connected layer for classification
        self.fc_final = nn.Linear(in_features=32*32*3, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Compress input features with global average pooling
        x = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)

        # Flatten the pooled features
        x = self.flatten(x)  # output tensor shape: (1, 3)

        # Generate weights using two fully connected layers
        weights = self.fc1(x)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)

        # Reshape weights to match input shape
        weights = weights.view(-1, 3, 32, 32)

        # Element-wise multiplication with input feature map
        x = x.view(-1, 3, 32, 32)
        x = x * weights

        # Flatten and pass through final fully connected layer
        x = self.flatten(x)
        x = self.fc_final(x)

        # Apply softmax to get probability distribution
        x = F.softmax(x, dim=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)