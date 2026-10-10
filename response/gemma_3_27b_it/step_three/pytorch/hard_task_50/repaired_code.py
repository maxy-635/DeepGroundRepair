import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=4032, out_features=128)

        # Block 2
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.fc2 = nn.Linear(in_features=32 * 4 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        pool1_out = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        pool2_out = self.pool2(x)  # output tensor shape: (1, 3, 16, 16)
        pool3_out = self.pool3(x)  # output tensor shape: (1, 3, 8, 8)

        pool1_flat = torch.flatten(pool1_out, start_dim=1)  # output tensor shape: (1, 3072)
        pool2_flat = torch.flatten(pool2_out, start_dim=1)  # output tensor shape: (1, 768)
        pool3_flat = torch.flatten(pool3_out, start_dim=1)  # output tensor shape: (1, 192)

        concat_input = torch.cat([pool1_flat, pool2_flat, pool3_flat], dim=1)  # output tensor shape: (1, 4032)
        dropout_out = self.dropout(concat_input)  # output tensor shape: (1, 4032)
        fc1_out = self.fc1(dropout_out)
        reshaped_out = torch.reshape(fc1_out, (-1, 128, 1, 1))

        # Block 2
        group1 = reshaped_out
        group2 = reshaped_out
        group3 = reshaped_out
        group4 = reshaped_out

        conv1_out = F.relu(self.conv1(group1))
        conv2_out = F.relu(self.conv2(group2))
        conv3_out = F.relu(self.conv3(group3))
        conv4_out = F.relu(self.conv4(group4))

        concat_out = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        flat_out = torch.flatten(concat_out, start_dim=1)
        output = self.fc2(flat_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)