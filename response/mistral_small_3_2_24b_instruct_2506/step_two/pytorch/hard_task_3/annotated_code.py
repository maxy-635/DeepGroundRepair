import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.conv1_1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.5)

        self.conv3_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.5)

        # Branch pathway
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=192 * 8 * 8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Process each group
        out1 = self.conv1_1(x1)
        out1 = self.conv1_2(out1)
        out1 = self.dropout1(out1)

        out2 = self.conv2_1(x2)
        out2 = self.conv2_2(out2)
        out2 = self.dropout2(out2)

        out3 = self.conv3_1(x3)
        out3 = self.conv3_2(out3)
        out3 = self.dropout3(out3)

        # Concatenate outputs
        main_out = torch.cat([out1, out2, out3], dim=1)

        # Branch pathway
        branch_out = self.branch_conv(x)

        # Combine pathways
        combined = torch.add(main_out, branch_out)

        # Flatten and pass through fully connected layer
        combined = combined.view(combined.size(0), -1)
        output = self.fc(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image size is 32x32
    output_tensor = model(input_tensor)
    print(output_tensor)