import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=1, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Global Average Pooling
        x = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)

        # Fully Connected Layer 1
        x = self.fc1(x)  # output tensor shape: (1, 3, 1, 32)

        # Reshape weights to align with input shape
        weights = torch.reshape(x, (1, 32, 1, 1))
        
        # Multiply weights with input feature map
        x = x.expand_as(x)
        x = x * weights

        # Flatten the result
        x = self.flatten(x)

        # Fully Connected Layer 2
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)