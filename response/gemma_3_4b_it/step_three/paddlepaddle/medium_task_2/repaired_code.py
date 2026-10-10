import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn_branch = nn.BatchNorm2D(num_features=16)

        # Combine features
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 16 * 16, out_features=128)
        self.bn_fc1 = nn.BatchNorm2D(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = nn.ReLU()(self.conv1(x))
        main_path = nn.ReLU()(self.bn1(main_path))
        main_path = nn.ReLU()(self.conv2(main_path))
        main_path = nn.ReLU()(self.bn2(main_path))
        main_path = self.pool(main_path)

        # Branch path
        branch_path = nn.ReLU()(self.conv_branch(x))
        branch_path = nn.ReLU()(self.bn_branch(branch_path))

        # Combine features
        combined_features = torch.cat((main_path, branch_path), dim=1)

        # Flatten and map to probability distribution
        flattened_features = self.flatten(combined_features)
        output = self.fc1(flattened_features)
        output = nn.ReLU()(self.bn_fc1(output))
        output = self.fc2(output)
        output = self.softmax(output)

        return output

if __name__ == "__main__":
    import torch
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)