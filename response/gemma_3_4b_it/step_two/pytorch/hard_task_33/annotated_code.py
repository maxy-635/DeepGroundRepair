import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 28, 28)
        output = self.block(x)
        output = torch.cat([output, output, output], dim=1)
        output = output.view(output.size(0), -1)
        output = nn.Linear(output.size(1), 10)(output)
        output = F.softmax(output, dim=1)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 28, 28)  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)