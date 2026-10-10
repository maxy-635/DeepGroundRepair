import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway
        self.main_pathway = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # Parallel pathway
        self.parallel_pathway = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        main_output = self.main_pathway(x)
        parallel_output = self.parallel_pathway(x)

        combined_output = torch.cat((main_output, parallel_output), dim=1)

        output = torch.add(combined_output, combined_output)

        output = nn.Linear(combined_output.size(-1), 10)(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32, 32)  # Batch size 1, 3 channels, 32x32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)