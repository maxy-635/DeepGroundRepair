import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=131072, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)  # output tensor shape: (1, 128, 32, 32)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 128, 32, 32)
        
        main_output = main_output.view(main_output.size(0), -1)  # output tensor shape: (1, 131072)
        branch_output = branch_output.view(branch_output.size(0), -1)  # output tensor shape: (1, 131072)
        
        combined_output = torch.add(main_output, branch_output)  # output tensor shape: (1, 131072)
        
        flattened_output = combined_output.view(combined_output.size(0), -1)  # output tensor shape: (1, 131072)
        output = self.fc(flattened_output)
        
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)