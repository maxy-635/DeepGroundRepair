import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel convolutional paths
        self.conv1x1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Transition Convolution
        self.trans_conv = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0)

        # Block 2: Global max pooling and fully connected layers
        self.global_max_pool = nn.MaxPool2d(kernel_size=32, stride=32, padding=0)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)

        # Final classification layer
        self.fc_final = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1: Three parallel convolutional paths
        path1 = self.conv1x1_1(x)
        path1 = self.conv3x3_1(path1)
        path1 = self.conv1x1_2(path1)

        path2 = self.conv1x1_3(x)
        path2 = self.conv3x3_2(path2)
        path2 = self.conv1x1_4(path2)

        path3 = self.conv1x1_5(x)
        path3 = self.conv3x3_3(path3)
        path3 = self.conv1x1_6(path3)

        # Concatenate the three paths
        concatenated = torch.cat((path1, path2, path3), dim=1)

        # Transition Convolution
        adjusted_output = self.trans_conv(concatenated)

        # Block 2: Global max pooling and fully connected layers
        pooled = self.global_max_pool(adjusted_output)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten

        fc1_out = self.fc1(pooled)
        fc2_out = self.fc2(fc1_out)

        # Reshape and multiply
        weights = fc2_out.view(-1, 128, 1, 1)
        main_path_output = torch.mul(adjusted_output, weights)

        # Branch connection
        branch_output = x
        final_output = main_path_output + branch_output

        # Final classification
        final_output = self.fc_final(final_output.view(final_output.size(0), -1))

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)