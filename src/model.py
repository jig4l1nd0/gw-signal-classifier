import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Use ConvTranspose1d for upsampling
        self.up = nn.ConvTranspose1d(in_channels,
                                     in_channels // 2,
                                     kernel_size=2,
                                     stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # Note: in_channels here

    def forward(self, x1, x2):
        """
        x1: input from the previous (upsampled) layer
        x2: input from the corresponding skip connection (encoder layer)
        """
        x1 = self.up(x1)

        # Concatenate skip connection (x2) with upsampled features (x1)
        # x2 will be from the DownBlock, x1 from the UpBlock
        # The concatenation happens along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """
    The main 1D U-Net model.
    It takes in a (Batch, 1, 2048) tensor 
    and outputs a (Batch, 1, 2048) tensor.
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 1. Encoder (Down-sampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        # 2. Decoder (Up-sampling path)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # 3. Output layer
        # Final 1x1 convolution to map to the desired number of classes (1, in our case)
        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, 1, 2048)

        # Encoder
        x1 = self.inc(x)     # -> (B, 64, 2048)
        x2 = self.down1(x1)  # -> (B, 128, 1024)
        x3 = self.down2(x2)  # -> (B, 256, 512)
        x4 = self.down3(x3)  # -> (B, 512, 256)
        x5 = self.down4(x4)  # -> (B, 1024, 128)

        # Decoder + Skip Connections
        x = self.up1(x5, x4)   # (B, 1024, 128) + (B, 512, 256) -> (B, 512, 256)
        x = self.up2(x, x3)    # (B, 512, 256)  + (B, 256, 512) -> (B, 256, 512)
        x = self.up3(x, x2)    # (B, 256, 512)  + (B, 128, 1024) -> (B, 128, 1024)
        x = self.up4(x, x1)    # (B, 128, 1024) + (B, 64, 2048) -> (B, 64, 2048)

        # Final output
        logits = self.outc(x)  # -> (B, 1, 2048)

        # We don't apply sigmoid here, as the loss function (BCEWithLogitsLoss)
        # is more numerically stable when it takes raw logits.
        return logits


# A quick test to ensure the model dimensions are correct
if __name__ == "__main__":
    # Create a dummy input tensor
    # (Batch size = 2, Channels = 1, Length = 2048)
    dummy_input = torch.randn(2, 1, 2048)

    # Initialize the model
    model = UNet1D()

    # Pass the dummy data through
    output = model(dummy_input)

    print("--- Model Test ---")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # The output shape should be the same as the input shape
    assert dummy_input.shape == output.shape

    print("Model test passed! Shapes match.")