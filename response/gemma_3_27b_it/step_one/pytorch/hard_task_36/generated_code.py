import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        main_pathway = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )

        # Branch Pathway
        branch_pathway = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)

        # Fusion
        self.fusion = nn.Add()

        # Classification Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32, out_features=10)

        self.main_pathway = main_pathway
        self.branch_pathway = branch_pathway

    def forward(self, x):
        main_output = self.main_pathway(x)
        branch_output = self.branch_pathway(x)

        fused_output = self.fusion(main_output, branch_output)

        pooled_output = self.global_avg_pool(fused_output)
        flattened_output = self.flatten(pooled_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)