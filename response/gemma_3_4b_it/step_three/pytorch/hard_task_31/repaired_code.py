import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        # Block 2
        self.block2_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.block2_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
        self.concat = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 32 * 32, 10)  # Assuming CIFAR-10

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        merged = torch.add(main_output, branch_output)

        block2_1 = self.block2_1(merged)
        block2_2 = self.block2_2(block2_1)
        block2_3 = self.block2_3(block2_2)
        concat_output = self.concat(block2_3)

        output = self.flatten(concat_output)
        output = self.fc(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)