import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block for each branch
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=64, out_features=32),  # Assuming input channels are 64
            nn.Linear(in_features=32, out_features=64)   # Output channels should match input
        )

        # Define the branches
        self.branch1 = nn.Sequential(
            self.block
        )

        self.branch2 = nn.Sequential(
            self.block
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=10)  # 128 because we concatenate two 64-channel outputs

    def forward(self, x):
        # Process through both branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        # Element-wise multiplication (assuming out1 and out2 are weights)
        out1 = torch.reshape(out1, x.shape)
        out2 = torch.reshape(out2, x.shape)
        out1 = x * out1
        out2 = x * out2

        # Concatenate the outputs
        out = torch.cat((out1, out2), dim=1)

        # Flatten and pass through final FC layer
        out = self.flatten(out)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor)