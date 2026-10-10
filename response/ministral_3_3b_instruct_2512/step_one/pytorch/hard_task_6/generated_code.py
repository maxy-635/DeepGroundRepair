# import torch
# import torch.nn as nn

# class DL_Model(nn.Module):
#     def __init__(self):
#         super(DL_Model, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
#         )

#         self.block2 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.ConvTranspose2d(3, 3, kernel_size=1, stride=1, padding=0),
#             nn.Permute((2, 0, 1)),
#             nn.ConvTranspose2d(3, 3, kernel_size=1, stride=1, padding=0)
#         )

#         self.block3 = nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=3, stride=1, depthwise=True)
#         )

#         self.fc = nn.Linear(3, 10)

#     def forward(self, x):
#         x1 = self.block1(x)
#         x2 = self.block2(x)
#         x3 = self.block3(x2)
#         x = torch.cat([x1, x2, x3], dim=1)
#         x = self.fc(x)
#         return x