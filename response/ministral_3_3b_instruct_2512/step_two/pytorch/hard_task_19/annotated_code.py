import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # Weight generation and addition
        self.weight_generation = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Final classification layers
        self.classification_layers = nn.Sequential(
            nn.Linear(in_features='mask', out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_output = self.main_path(x)  # output tensor shape: (1, 128, 16, 16)

        # Branch path
        branch_output = self.branch_path(main_output)  # output tensor shape: (1, 128)

        # Generate weights
        weights = self.weight_generation(branch_output)  # output tensor shape: (1, 128)

        # Reshape and multiply
        weights = weights.view(weights.size(0), 128, 1, 1)  # output tensor shape: (1, 128, 1, 1)
        weighted_main = torch.add(main_output, weights)  # output tensor shape: (1, 128, 16, 16)

        # Classification
        output = self.classification_layers(weighted_main)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)