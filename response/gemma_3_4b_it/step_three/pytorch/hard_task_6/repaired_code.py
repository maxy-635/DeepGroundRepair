import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv1d(3, 32, 1, 1, 0)
        self.block1_conv2 = nn.Conv1d(32, 16, 1, 1, 0)
        self.block1_conv3 = nn.Conv1d(16, 16, 1, 1, 0)
        self.block1_relu = nn.ReLU(inplace=True)
        self.block1_concat = nn.Concat([self.block1_conv1(torch.randn(1, 3, 32)),
                                        self.block1_conv2(torch.randn(1, 32, 16)),
                                        self.block1_conv3(torch.randn(1, 16, 16))])
        self.block1_relu = nn.ReLU(inplace=True)

        # Block 2
        self.block2_reshape = nn.Reshape((32, 32, 3, 16))
        self.block2_permute = nn.Permute([2, 3, 1, 0])
        self.block2_reshape2 = nn.Reshape((32, 32, 16, 3))
        self.block2_relu = nn.ReLU(inplace=True)

        # Block 3
        self.block3_conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.block3_relu = nn.ReLU(inplace=True)

        # Branch
        self.branch_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.branch_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Main path
        main_path = self.block1_concat
        main_path = self.block2_reshape(main_path)
        main_path = self.block2_permute(main_path)
        main_path = self.block2_reshape2(main_path)
        main_path = self.block3_conv(main_path)
        main_path = self.block3_relu(main_path)

        # Branch path
        branch_path = self.branch_pool(x)
        branch_path = self.branch_avg_pool(branch_path)
        branch_path = torch.flatten(branch_path, 1)

        # Concatenate
        combined = torch.cat([main_path, branch_path], dim=1)

        # Fully connected layer
        output = self.fc(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)