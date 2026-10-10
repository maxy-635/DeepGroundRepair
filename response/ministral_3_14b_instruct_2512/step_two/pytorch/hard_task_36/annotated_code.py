import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask'),
            nn.Dropout2d(p=0.5)
        )

        # Branch pathway
        self.branch_pathway = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main pathway processing
        main_out = self.main_pathway(x)  # output tensor shape of this layer should be (1, 32, 28, 28)

        # Branch pathway processing
        branch_out = self.branch_pathway(x)  # output tensor shape: (1, 32, 28, 28)

        # Fusion
        fused_out = main_out + branch_out
        fused_out = self.fusion(fused_out)

        # Final layers
        pooled_out = self.global_avg_pool(fused_out)
        flattened_out = self.flatten(pooled_out)
        output = self.fc(flattened_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)