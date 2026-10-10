import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Stage 1: Convolution and Max Pooling
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Stage 2: Convolution and Dropout
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(num_features=256)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(num_features=512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.25)

        # Stage 3: Upsampling with Skip Connections
        self.conv_up1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.batchnorm_up1 = nn.BatchNorm2d(num_features=256)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv_up2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchnorm_up2 = nn.BatchNorm2d(num_features=128)
        self.relu6 = nn.ReLU(inplace=True)
        
        # Final Convolutional Layer
        self.conv_final = nn.Conv2d(in_channels=128, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Stage 1
        x = self.conv1_1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv1_2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Stage 2
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.relu4(x)
        x = self.dropout2(x)

        # Stage 3
        x = self.conv_up1_1(x)
        x = self.batchnorm_up1(x)
        x = self.relu5(x)
        
        x = self.conv_up2_1(x)
        x = self.batchnorm_up2(x)
        x = self.relu6(x)

        # Final Layer
        x = self.conv_final(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 images are 32x32)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: torch.Size([1, 10, 32, 32])