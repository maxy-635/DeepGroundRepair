import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*7*7, out_features=10)  # Assuming input size is 28x28 after two max pooling layers

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_output = self.main_path(x)  # output tensor shape: (1, 64, 7, 7)

        # Branch path
        branch_output = self.branch_path(x)  # output tensor shape: (1, 32, 28, 28)

        # Sum the outputs
        combined_output = torch.add(main_output, branch_output)

        # Flatten and pass through fully connected layer
        flattened_output = self.flatten(combined_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)