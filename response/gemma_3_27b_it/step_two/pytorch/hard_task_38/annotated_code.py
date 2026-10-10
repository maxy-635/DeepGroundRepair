import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the repeated block
        block = nn.Sequential(  # output tensor shape: (unknown)
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Pathway 1
        self.pathway1 = nn.Sequential(*[block for _ in range(3)])  # output tensor shape: (1)

        # Pathway 2
        self.pathway2 = nn.Sequential(*[block for _ in range(3)])  # output tensor shape: (1)

        # Merge pathways
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification layers
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Process through pathways
        pathway1_output = self.pathway1(x)  # output tensor shape: (1, 1, 28, 28)
        pathway2_output = self.pathway2(x)  # output tensor shape: (1, 1, 28, 28)

        # Concatenate pathways
        concatenated = torch.concatenate([pathway1_output, pathway2_output], dim=1)  # output tensor shape: (1, 2, 28, 28)

        # Merge and classify
        merged = self.concat(concatenated)  # output tensor shape: (1, 16, 14, 14)
        merged = torch.flatten(merged, start_dim=1)  # output tensor shape: (1, 3136)
        output = self.fc1(merged)
        output = self.relu(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)