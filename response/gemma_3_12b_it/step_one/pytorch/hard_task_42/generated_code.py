import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_path1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.block1_path2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.block1_path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )
        self.block1_concat = nn.Sequential(
            nn.Cat()
        )
        self.block1_fc = nn.Linear(in_features=136, out_features=128)
        self.block1_reshape = nn.Sequential(
            nn.Reshape(shape=(1, 128))
        )

        # Block 2
        self.block2_path1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.block2_path2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        )
        self.block2_path3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        )
        self.block2_path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_concat = nn.Sequential(
            nn.Cat()
        )

        # Classifier
        self.classifier_fc1 = nn.Linear(in_features=128, out_features=64)
        self.classifier_fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        out1_path1 = self.block1_path1(x)
        out1_path2 = self.block1_path2(x)
        out1_path3 = self.block1_path3(x)
        out1_concat = torch.cat((out1_path1, out1_path2, out1_path3), dim=1)
        out1_fc = self.block1_fc(out1_concat)
        out1_reshape = self.block1_reshape(out1_fc)

        # Block 2
        out2_path1 = self.block2_path1(out1_reshape)
        out2_path2 = self.block2_path2(out1_reshape)
        out2_path3 = self.block2_path3(out1_reshape)
        out2_path4 = self.block2_path4(out1_reshape)
        out2_concat = torch.cat((out2_path1, out2_path2, out2_path3, out2_path4), dim=1)

        # Classifier
        out_classifier_fc1 = self.classifier_fc1(out2_concat.squeeze())
        out_classifier_fc2 = self.classifier_fc2(out_classifier_fc1)

        return out_classifier_fc2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)