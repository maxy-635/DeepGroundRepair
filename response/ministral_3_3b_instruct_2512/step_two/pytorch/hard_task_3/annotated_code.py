import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Branch pathway
        self.branch_pathway = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layer
        self.fc = nn.Linear(64 * 8 * 8, 10)  # Assuming input size is 32x32, after pooling, 8x8

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        x1, x2, x3 = torch.split(x, 64, dim=1)

        # Process each group through the main pathway
        out1 = self.main_pathway(x1)
        out2 = self.main_pathway(x2)
        out3 = self.main_pathway(x3)

        # Concatenate the outputs from the three groups
        main_output = torch.cat((out1, out2, out3), dim=1)

        # Process the branch pathway
        branch_output = self.branch_pathway(x)

        # Combine the outputs from both pathways using addition
        combined_output = torch.add(main_output, branch_output)

        # Apply adaptive average pooling and flatten
        combined_output = nn.AdaptiveAvgPool2d((8, 8))(combined_output)
        combined_output = combined_output.view(combined_output.size(0), -1)

        # Pass through the fully connected layer
        output = self.fc(combined_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)