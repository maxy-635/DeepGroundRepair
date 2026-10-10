import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel average pooling paths
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer and reshaping between blocks
        self.fc1 = nn.Linear(28*28*3, 1024)  # Assuming input is 28x28 with 3 channels after pooling
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 16*16*64)  # Reshape to 16x16x64 for block 2
        )

        # Block 2: Four parallel branches
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch 2: 1x1 + 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: 1x1 + 3x3 + 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 4: Average pooling + 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final classification layers
        self.fc2 = nn.Linear(64*16*16, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 classes for MNIST

    def forward(self, x):
        # Block 1 processing
        pool1 = self.avg_pool1(x)
        pool2 = self.avg_pool2(x)
        pool3 = self.avg_pool3(x)

        # Flatten and apply dropout
        pool1 = pool1.view(pool1.size(0), -1)
        pool2 = pool2.view(pool2.size(0), -1)
        pool3 = pool3.view(pool3.size(0), -1)

        pool1 = self.dropout(pool1)
        pool2 = self.dropout(pool2)
        pool3 = self.dropout(pool3)

        # Concatenate the three paths
        block1_out = torch.cat([pool1, pool2, pool3], dim=1)

        # Transform to 4D tensor for block 2
        block1_out = self.fc1(block1_out)
        block1_out = self.reshape(block1_out)
        block1_out = block1_out.view(-1, 64, 16, 16)

        # Block 2 processing
        branch1_out = self.branch1(block1_out)
        branch2_out = self.branch2(block1_out)
        branch3_out = self.branch3(block1_out)
        branch4_out = self.branch4(block1_out)

        # Concatenate all branches
        block2_out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

        # Final classification
        block2_out = block2_out.view(block2_out.size(0), -1)
        block2_out = self.fc2(block2_out)
        block2_out = F.relu(block2_out)
        block2_out = self.fc3(block2_out)

        return block2_out

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel, but we need to adjust for our architecture
    # We'll create a dummy input that matches what our model expects after block 1 processing
    # The model expects input to be 28x28x1 (MNIST) but transforms it internally
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10] for 10 classes