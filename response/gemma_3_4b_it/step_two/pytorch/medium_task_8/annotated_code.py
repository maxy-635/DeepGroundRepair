import torch
import torch.nn as nn
import torch.utils.data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.main_path_concat = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.main_path_concat_final = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.main_path_concat_final = nn.ReLU(inplace=True)
        

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Add()

        # Classification
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, 10)  # Output for CIFAR-10

    def forward(self, x):
        main_path_group1 = x
        main_path_group2 = self.main_path(x)
        main_path_group3 = self.main_path_concat_final(x)
        
        combined_main_path = torch.cat((main_path_group1, main_path_group2, main_path_group3), dim=1)
        
        branch_path_output = self.branch_path(x)
        
        combined_output = self.fusion(combined_main_path, branch_path_output)
        
        output = self.flatten(combined_output)
        output = self.linear(output)
        
        return output


if __name__ == "__main__":
    # Example Usage:
    model = DL_Model()
    
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected output: torch.Size([1, 10])