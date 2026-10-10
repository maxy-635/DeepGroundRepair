import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(in_features=32 * 32 * 32, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        weights = self.fc1(x)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = weights.reshape(x.shape)
        x = x * weights
        x = self.flatten(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)