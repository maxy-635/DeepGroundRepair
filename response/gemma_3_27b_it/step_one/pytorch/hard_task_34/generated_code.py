import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        main_path_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1)
        )

        self.main_path = nn.Sequential(
            main_path_block,
            nn.Conv2d(in_channels=17, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1),
            main_path_block,
            nn.Conv2d(in_channels=33, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1),
            main_path_block,
            nn.Conv2d(in_channels=65, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1)
        )

        # Branch Path
        self.branch_path = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, groups=1)

        # Fusion and Classification
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=10)

    def forward(self, x):
        # Main Path
        main_path_input = x
        main_path_output = self.main_path[0](main_path_input)
        main_path_output = torch.cat((main_path_input, main_path_output), dim=1)
        
        main_path_output = self.main_path[1](main_path_output)
        main_path_output = torch.cat((main_path_output, self.main_path[2](main_path_output)), dim=1)

        main_path_output = self.main_path[3](main_path_output)
        main_path_output = torch.cat((main_path_output, self.main_path[4](main_path_output)), dim=1)

        main_path_output = self.main_path[5](main_path_output)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Fusion
        fused_features = torch.add(main_path_output, branch_path_output)

        # Classification
        flattened_features = self.flatten(fused_features)
        output = self.fc(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)