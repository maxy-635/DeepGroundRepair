import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_group1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = F.dropout(torch.randn(1, 32, 32, 32), p=0.5)

        self.conv1_group2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = F.dropout(torch.randn(1, 32, 32, 32), p=0.5)

        self.conv1_group3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_group3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = F.dropout(torch.randn(1, 32, 32, 32), p=0.5)

        self.branch_conv = nn.Conv2d(in_channels=3, out_channels='mask', kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main pathway
        group1 = self.conv1_group1(x)  # output tensor shape: (1, 32, 32, 32)
        group1 = self.conv2_group1(group1)  # output tensor shape: (1, 64, 32, 32)
        group1 = self.dropout1  # output tensor shape: (1, 32, 32, 32)
        group2 = self.conv1_group2(x)  # output tensor shape: (1, 32, 32, 32)
        group2 = self.conv2_group2(group2)  # output tensor shape: (1, 64, 32, 32)
        group2 = self.dropout2  # output tensor shape: (1, 32, 32, 32)
        group3 = self.conv1_group3(x)  # output tensor shape: (1, 32, 32, 32)
        group3 = self.conv2_group3(group3)  # output tensor shape: (1, 64, 32, 32)
        group3 = self.dropout3  # output tensor shape: (1, 32, 32, 32)
        main_pathway = torch.cat((group1, group2, group3), dim=1)  # output tensor shape: (1, 96, 32, 32)

        # Branch pathway
        branch_pathway = self.branch_conv(x)  # output tensor shape of this layer should be (1, 96, 32, 32)

        # Combine pathways
        combined_pathway = torch.add(main_pathway, branch_pathway)

        # Flatten and classify
        flattened = combined_pathway.view(combined_pathway.size(0), -1)
        output = self.fc(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)