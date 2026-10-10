import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three average pooling layers
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=4032, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply the average pooling layers
        out1 = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        out2 = self.pool2(x)  # output tensor shape: (1, 3, 16, 16)
        out3 = self.pool3(x)  # output tensor shape: (1, 3, 8, 8)

        # Flatten the outputs of the pooling layers
        flat1 = torch.flatten(out1, start_dim=1)  # output tensor shape: (1, 3072)
        flat2 = torch.flatten(out2, start_dim=1)  # output tensor shape: (1, 768)
        flat3 = torch.flatten(out3, start_dim=1)  # output tensor shape: (1, 192)

        # Concatenate the flattened features
        concatenated_features = torch.concat([flat1, flat2, flat3], dim=1)  # output tensor shape: (1, 4032)

        # Pass through the fully connected layers
        x = self.fc1(concatenated_features)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 color channels
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: torch.Size([1, 10])