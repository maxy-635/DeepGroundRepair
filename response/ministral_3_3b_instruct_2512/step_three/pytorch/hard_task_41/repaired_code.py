import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with average pooling of different scales
        self.pooling1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.pooling2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pooling3 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate the outputs of the three paths
        self.concat_block1 = nn.Sequential(
            nn.Linear(in_features=1029, out_features=128),
            nn.ReLU()
        )

        # Block 2: Multiple branch connections for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.ReLU()
        )

        # Concatenate the outputs of the four branches
        self.concat_block2 = nn.Sequential(
            nn.Linear(128*4, 256),  # Adjust dimensions based on actual output shapes
            nn.ReLU()
        )

        # Final classification layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths
        pool1 = self.pooling1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.pooling2(x)  # output tensor shape: (1, 1, 14, 14)
        pool3 = self.pooling3(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and dropout
        flat1 = self.flatten1(pool1)  # output tensor shape: (1, 784)
        flat2 = self.flatten2(pool2)  # output tensor shape: (1, 196)
        flat3 = self.flatten3(pool3)  # output tensor shape: (1, 49)

        dropout1 = self.dropout1(flat1)  # output tensor shape: (1, 784)
        dropout2 = self.dropout2(flat2)  # output tensor shape: (1, 196)
        dropout3 = self.dropout3(flat3)  # output tensor shape: (1, 49)

        # Concatenate outputs
        concat_block1 = torch.cat((dropout1, dropout2, dropout3), dim=1)  # output tensor shape: (1, 1029)
        concat_block1 = self.concat_block1(concat_block1)

        # Reshape to 4D tensor for block 2
        reshaped = torch.reshape(concat_block1, (concat_block1.size(0), 1, 128, 1))

        # Block 2: Multiple branches
        branch1_out = self.branch1(reshaped)
        branch2_out = self.branch2(reshaped)
        branch3_out = self.branch3(reshaped)
        branch4_out = self.branch4(reshaped)

        # Concatenate outputs of branches
        branch_concat = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)
        branch_concat = self.concat_block2(branch_concat)

        # Final classification layers
        fc1_out = self.fc1(branch_concat)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    # Create a dummy input tensor for MNIST (1 channel, 28x28)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)