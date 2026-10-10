import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(32, 32), scale_factor=2, mode='bilinear'),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(32, 32), scale_factor=2, mode='bilinear'),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Concatenate branches
        self.concat_layer = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Final linear layer
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        concatenated = torch.cat((main_output, branch1_output, branch2_output, branch3_output), dim=1)
        final_output = self.concat_layer(concatenated)

        output = self.fc1(final_output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)