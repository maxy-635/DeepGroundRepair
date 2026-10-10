import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(3, 10)

    def forward(self, x):
        # Block 1
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_concat = torch.cat((x1, x2, x3), dim=1)

        # Block 2
        shape = x_concat.shape
        height, width, groups, channels_per_group = shape[1], shape[2], shape[0], shape[3]
        x_reshaped = torch.reshape(x_concat, (height, width, groups, channels_per_group))
        x_permuted = torch.permute(x_reshaped, (0, 2, 1, 3))
        x_reshaped_back = torch.reshape(x_permuted, (height, width, channels_per_group * groups))

        # Block 3
        x_depthwise = self.conv5(x_reshaped_back)

        # Branch
        branch = x

        # Combine
        combined = torch.add(x_depthwise, branch)

        # Fully connected layer
        output = self.linear(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Example input shape: batch_size, channels, width
    output_tensor = model(input_tensor)
    print(output_tensor.shape)