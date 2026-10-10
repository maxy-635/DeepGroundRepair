import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the channel dimension
        self.group1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        )

        # Combine the outputs from the three groups
        self.add_group1 = nn.Sequential(
            nn.Add()
        )
        self.add_group2 = nn.Sequential(
            nn.Add()
        )
        self.add_group3 = nn.Sequential(
            nn.Add()
        )

        # Combine the main path with the original input
        self.add_main_input = nn.Sequential(
            nn.Add()
        )

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 32, 10)  # Adjust input features based on your input size

    def forward(self, x):
        group1_out = self.group1(x)
        group2_out = self.group2(x)
        group3_out = self.group3(x)

        # Combine the outputs from the three groups
        combined_group1 = self.add_group1(group1_out)
        combined_group2 = self.add_group2(group2_out)
        combined_group3 = self.add_group3(group3_out)

        # Combine the main path with the original input
        main_path = torch.add(combined_group1, combined_group2, combined_group3)
        main_path = torch.add(main_path, x)

        # Flatten and pass through fully connected layer
        flattened = self.flatten(main_path)
        output = self.fc(flattened)
        return output

if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)