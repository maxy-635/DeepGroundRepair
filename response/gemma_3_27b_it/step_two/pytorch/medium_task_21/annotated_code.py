import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Concatenate the outputs from all branches
        self.concat = nn.Sequential(
            branch1,
            branch2,
            branch3,
            branch4
        )
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 4 * 32 * 32, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # Pass input through each branch
        branch1_out = self.concat[0](x)
        branch2_out = self.concat[1](x)
        branch3_out = self.concat[2](x)
        branch4_out = self.concat[3](x)

        # Concatenate the outputs
        concat_out = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)

        # Flatten the concatenated output
        flattened_out = concat_out.view(concat_out.size(0), -1)

        # Pass through fully connected layers
        output = self.fc(flattened_out)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)