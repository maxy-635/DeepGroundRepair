import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        main_path_output = self.main_path(x)  # output tensor shape: (1, 128, 28, 28)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 128, 28, 28)
        combined_output = torch.add(main_path_output, branch_path_output)  # output tensor shape: (1, 128, 28, 28)
        flattened_output = self.flatten(combined_output)  # output tensor shape: (1, 100352)
        output = self.linear(flattened_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)