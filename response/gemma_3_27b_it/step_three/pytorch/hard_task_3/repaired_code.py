import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.main_pathway2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.main_pathway3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Branch Pathway
        self.branch_pathway = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)  # Assuming input image size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Pathway
        main1 = self.main_pathway(x)  # output tensor shape: (1, 64, 32, 32)
        main2 = self.main_pathway2(x)  # output tensor shape: (1, 64, 32, 32)
        main3 = self.main_pathway3(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate Main Pathway outputs
        main_output = torch.cat([main1, main2, main3], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Branch Pathway
        branch_output = self.branch_pathway(x)  # output tensor shape of this layer should be (1, 192, 32, 32)

        # Combine Pathways
        combined_output = main_output + branch_output

        # Flatten and Fully Connect
        flattened_output = combined_output.view(combined_output.size(0), -1)
        output = self.fc(flattened_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])