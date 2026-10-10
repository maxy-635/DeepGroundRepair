import torch
import torch.nn.functional as F

class DL_Model(torch.nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.linear1 = torch.nn.Linear(32, 64)  # Example linear layer
        self.linear2 = torch.nn.Linear(64, 10) # Example linear layer

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 3)
        x = torch.reshape(x, (x.shape[0], -1))  # output tensor shape: (1, 3072)
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, x.shape[0], 32, 32, 3)
        x = F.softmax(self.linear2(F.relu(self.linear1(x))), dim=1)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 32, 32, 3)  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)