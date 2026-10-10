import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1
        self.conv1_main = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_branch = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Block 2
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=1029, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        main_path = self.relu1(self.conv1_main(x))  # output tensor shape: (1, 16, 28, 28)
        branch_path = self.relu1(self.conv1_branch(x))  # output tensor shape: (1, 16, 28, 28)
        combined_path = main_path + branch_path  # output tensor shape: (1, 16, 28, 28)
        conv2_out = self.conv2(combined_path)  # output tensor shape: (1, 1, 28, 28)
        
        # Block 2
        pool1_out = self.pool1(conv2_out)  # output tensor shape: (1, 1, 28, 28)
        pool2_out = self.pool2(conv2_out)  # output tensor shape: (1, 1, 14, 14)
        pool3_out = self.pool3(conv2_out)  # output tensor shape: (1, 1, 7, 7)
        
        flattened_pool1 = self.flatten(pool1_out)  # output tensor shape: (1, 784)
        flattened_pool2 = self.flatten(pool2_out)  # output tensor shape: (1, 196)
        flattened_pool3 = self.flatten(pool3_out)  # output tensor shape: (1, 49)

        concatenated = torch.cat((flattened_pool1, flattened_pool2, flattened_pool3), dim=1)  # output tensor shape: (1, 1029)

        # Fully Connected Layers
        fc1_out = torch.relu(self.fc1(concatenated))
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input: batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)