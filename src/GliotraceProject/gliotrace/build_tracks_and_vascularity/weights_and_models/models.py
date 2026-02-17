import torch
import torch.nn as nn
import torch.nn.functional as F


class StateNet(nn.Module):
    """
    6-state network
    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """

    def __init__(self, num_classes: int, emb_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-5)

        # For 61x61: 61 -> 31 -> 16 -> 8
        self.fc1 = nn.Linear(8 * 8 * 256, 512)

        # Embedding (NO ReLU here)
        self.fc2 = nn.Linear(512, emb_dim)

        # Standard classifier head
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        emb = self.fc2(x)
        logits = self.classifier(emb)

        if return_embedding:
            return logits, emb
        return logits


class TMENet(nn.Module):
    """
    TME network
    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """

    def __init__(self):
        super().__init__()

        # 2: conv1 – 32 filters, 3x3, stride 1, padding 'same'
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 4: bn1 – 32 channels
        self.bn1 = nn.BatchNorm2d(32)

        # 5: pool1 – 2x2, stride 2, padding 'same'
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 6: conv2 – 64 filters, 3x3
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 8: bn2 – 64 channels
        self.bn2 = nn.BatchNorm2d(64)

        # 9: pool2 – 2x2, stride 2, 'same'
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 10: conv3 – 128 filters, 3x3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 12: bn3 – 128 channels
        self.bn3 = nn.BatchNorm2d(128)

        # 13: pool3 – 2x2, stride 2, 'same'
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 14: conv4 – 256 filters, 3x3
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 16: bn4 – 256 channels
        self.bn4 = nn.BatchNorm2d(256)

        # 17: fc1 – 512 units
        self.fc1 = nn.Linear(8 * 8 * 256, 512)

        # 19: fc2 – 256 units
        self.fc2 = nn.Linear(512, 256)

        # 21: fc3 – 2 units (final output)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        # x: [N, 3, 61, 61]

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


class VascNet(nn.Module):
    """
    Vascular Segementation Network
    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.drop4 = nn.Dropout2d(p=0.3)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        # Decoder
        self.upsample3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.conv5_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        # Final 1×1 conv → 2 channels (classes)
        self.conv_final = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [N, 1, H, H]

        # Encoder
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)  # H -> H/2

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)  # H/2 -> H/4

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)  # H/4 -> H/8

        x = F.relu(self.conv4_1(x))
        x = self.drop4(x)
        x = F.relu(self.conv4_2(x))  # still H/8×H/8

        # Decoder
        x = self.upsample3(x)  # H/8 -> H/4
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))

        x = self.upsample2(x)  # H/4 -> H/2
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))

        x = self.upsample1(x)  # H/2 -> H
        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))

        x = self.conv_final(x)  # [N, 2, H, H] logits

        return x
