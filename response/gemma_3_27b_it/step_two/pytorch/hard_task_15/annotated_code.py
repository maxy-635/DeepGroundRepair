import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=32),
        )

        # Branch Path (Direct connection)

        # Combined layers
        self.combined_layers = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path_output = self.main_path(x)
        main_path_output = torch.reshape(main_path_output, (x.shape[0], 32, 1, 1))
        main_path_output = torch.mul(x, main_path_output)

        # Branch Path
        branch_path_output = x

        # Combine Paths
        combined_output = torch.cat((main_path_output, branch_path_output), dim=1)

        # Combined Layers
        final_output = self.combined_layers(combined_output)
        return final_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)