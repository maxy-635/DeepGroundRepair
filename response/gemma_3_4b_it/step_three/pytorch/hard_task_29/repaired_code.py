import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.main_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        )
        self.block1_merge = nn.Add()

        # Second block
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        self.flatten = nn.Flatten()
        self.concat = nn.Concat([self.pool1(self.main_path(torch.rand(1, 1, 28, 28))),
                                self.pool2(self.branch_path(torch.rand(1, 1, 28, 28))),
                                self.pool3(self.main_path(torch.rand(1, 1, 28, 28)))])

        # Fully connected layers
        self.fc1 = nn.Linear(self.concat.shape[1], 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        block1_output = self.block1_merge(main_path_output, branch_path_output)

        pool1_output = self.pool1(block1_output)
        pool2_output = self.pool2(block1_output)
        pool3_output = self.pool3(block1_output)

        flattened_output = self.flatten(block1_output)
        concatenated_output = self.concat(torch.cat([pool1_output, pool2_output, pool3_output], dim=1))

        fc1_output = torch.relu(self.fc1(concatenated_output))
        fc2_output = self.fc2(fc1_output)

        return fc2_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.rand(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)