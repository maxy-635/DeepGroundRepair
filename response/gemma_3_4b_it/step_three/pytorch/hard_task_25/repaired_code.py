import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Branch paths
        self.branch1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )


        # Concatenation
        self.concat_layer = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32)
        main_path_output = self.main_path(x)  # output tensor shape: (1, 64, 32)
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)

        concatenated = torch.cat((main_path_output, branch1_output, branch2_output, branch3_output), dim=1)
        
        final_output = self.concat_layer(concatenated)
        
        output = F.relu(final_output)
        output = output.view(-1, 10)
        final_output = self.fc(output)

        return final_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32)  # Example input shape: (batch_size, channels, spatial_size)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)