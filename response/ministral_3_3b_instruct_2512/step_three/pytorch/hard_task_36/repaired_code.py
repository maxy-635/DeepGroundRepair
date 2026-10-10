import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Fixed padding to 1
            nn.Conv2d(32, 32, kernel_size=1, padding=0),  # Fixed padding to 0
            nn.Conv2d(32, 32, kernel_size=1, padding=0),  # Fixed padding to 0
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Fixed padding to 1
            nn.Dropout(0.5)  # 50% dropout
        )

        # Branch pathway
        self.branch_pathway = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Fixed padding to 1
        )

        # Fusion and final layers
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, padding=0)  # Concatenate main and branch outputs
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        main_output = self.main_pathway(x)
        branch_output = self.branch_pathway(x)

        # Concatenate along channel dimension
        fused_output = torch.cat((main_output, branch_output), dim=1)

        # Apply fusion layer
        fused_output = self.fusion(fused_output)

        # Final layers
        pooled_output = self.avg_pool(fused_output)
        flattened_output = self.flatten(pooled_output)
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)