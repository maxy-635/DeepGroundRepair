import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block
        self.main_path_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

        # Repeat the main path block three times
        self.main_path = nn.Sequential('mask', 'mask', 'mask')





        # Branch path
        self.branch_path = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_features = self.main_path(x)

        # Branch path
        branch_features = self.branch_path(x)

        # Concatenate along channel dimension
        concatenated_features = torch.cat((main_features, branch_features), dim=1)

        # Flatten the features
        flattened_features = torch.flatten(concatenated_features, start_dim=1)

        # Fully connected layer
        output = self.fc(flattened_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)