import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_path1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.5)
        )
        self.block1_path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.5)
        )
        self.block1_path3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.5)
        )
        self.block1_concat = nn.Sequential(
            nn.Cat(dim=1)
        )
        self.block1_fc = nn.Linear(in_features=1024, out_features=128) # Assuming flattened size after concat is 1024, adjust as needed
        self.block1_reshape = nn.Reshape(shape=(4, 32)) # Reshape to 4x32, adjust dimensions as needed.

        # Block 2
        self.block2_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.block2_branch3 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.block2_branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_concat = nn.Sequential(
            nn.Cat(dim=1)
        )

        # Final Classification
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        path1 = self.block1_path1(x)
        path2 = self.block1_path2(x)
        path3 = self.block1_path3(x)

        concat1 = self.block1_concat(path1, path2, path3)
        fc1_out = self.block1_fc(concat1)
        reshaped_out = self.block1_reshape(fc1_out)

        # Block 2
        branch1 = self.block2_branch1(reshaped_out)
        branch2 = self.block2_branch2(reshaped_out)
        branch3 = self.block2_branch3(reshaped_out)
        branch4 = self.block2_branch4(reshaped_out)
        concat2 = self.block2_concat(branch1, branch2, branch3, branch4)

        # Final Classification
        fc1_out = self.fc1(concat2)
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example MNIST input: batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape) # Expected output shape: torch.Size([1, 10])