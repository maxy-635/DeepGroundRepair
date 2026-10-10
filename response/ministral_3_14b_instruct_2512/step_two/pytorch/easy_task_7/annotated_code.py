import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=10)
        )

    def forward(self, x):
        # Main path output
        main_out = self.main_path(x)

        # Branch path output
        branch_out = self.branch_path(x)

        # Combine outputs
        combined = torch.add(main_out, branch_out)

        # Classification
        output = self.classifier(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)