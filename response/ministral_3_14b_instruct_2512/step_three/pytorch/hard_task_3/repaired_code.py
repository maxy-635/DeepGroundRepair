import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.split_conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.split_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.split_conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.split_conv4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.split_conv5 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.split_conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        # Branch pathway
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Concatenation and final layers
        self.fc = nn.Linear(in_features=32*32*9, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups along channel dimension
        x1, x2, x3 = x.chunk(3, dim=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Main pathway
        # Group 1
        x1 = self.split_conv1(x1)
        x1 = F.relu(x1)
        x1 = self.split_conv2(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        # Group 2
        x2 = self.split_conv3(x2)
        x2 = F.relu(x2)
        x2 = self.split_conv4(x2)
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)

        # Group 3
        x3 = self.split_conv5(x3)
        x3 = F.relu(x3)
        x3 = self.split_conv6(x3)
        x3 = F.relu(x3)
        x3 = self.dropout3(x3)

        # Concatenate the three groups
        x_main = torch.cat([x1, x2, x3], dim=1)

        # Branch pathway
        x_branch = self.branch_conv(x)
        x_branch = F.relu(x_branch)

        # Combine main and branch pathways
        x_combined = x_main + x_branch

        # Flatten and pass through fully connected layer
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_out = self.fc(x_combined)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)