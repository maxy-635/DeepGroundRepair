import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three sequential blocks
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        block2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        block3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Define the parallel branch
        parallel_branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        # Combine the outputs
        self.combined_layers = nn.Sequential(
            block1,
            block2,
            block3,
            parallel_branch
        )

        # Flatten the output
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 + 32 + 64 + 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pass input through each block and parallel branch
        out1 = self.combined_layers[0](x)
        out2 = self.combined_layers[1](x)
        out3 = self.combined_layers[2](x)
        out_parallel = self.combined_layers[3](x)

        # Add the outputs
        combined_output = out1 + out2 + out3 + out_parallel

        # Flatten the combined output
        flattened_output = self.flatten(combined_output)

        # Pass through fully connected layers
        fc1_output = torch.relu(self.fc1(flattened_output))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)