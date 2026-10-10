import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1
        branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 2
        branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 3
        branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.2)
        )

        # Concatenate branches
        self.concat = nn.Sequential(
            nn.Concat(dim=1, tensors=[branch1(torch.randn(1,3,32,32)), branch2(torch.randn(1,3,32,32)), branch3(torch.randn(1,3,32,32))]),
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*8*8, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # Process through branches
        branch1_out = self.concat[0][0](x)
        branch2_out = self.concat[0][1](x)
        branch3_out = self.concat[0][2](x)
        
        # Concatenate outputs
        combined = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Flatten and pass through fully connected layers
        flattened = combined.view(combined.size(0), -1)
        output = self.fc(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)