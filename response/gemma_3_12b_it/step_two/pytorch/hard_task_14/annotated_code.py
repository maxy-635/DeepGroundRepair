import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main Path
        self.global_avg_pool = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.fc1 = nn.Linear(in_features=1280, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=3072)  # Match input channels (32*32*3 = 3072)

        # Branch Path
        self.conv_branch = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Final Layers
        self.fc3 = nn.Linear(in_features=3072, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        main_path = self.global_avg_pool(x)  # output tensor shape: (1, 3, 11, 11)
        main_path = torch.reshape('mask', 'mask')
        main_path = F.relu(self.fc1(main_path))
        main_path = F.relu(self.fc2(main_path))
        main_path = main_path.reshape((-1, 3, 32, 32))
        
        # Branch Path
        branch_path = self.conv_branch(x)

        # Add the outputs of both paths
        combined_path = torch.add(main_path, branch_path)

        # Final Layers
        combined_path = torch.reshape(combined_path, (-1, 3072))
        combined_path = F.relu(self.fc3(combined_path))
        output = self.fc4(combined_path)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)