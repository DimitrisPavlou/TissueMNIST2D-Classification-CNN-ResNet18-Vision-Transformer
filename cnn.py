import torch.nn as nn 


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers, batch normalization, and optional max pooling.
    """
    def __init__(self, in_channels, mid_channels, out_channels, max_pool=False):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        
        self.max_pool = max_pool
        if max_pool:
            self.pooling = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.conv_block(x)
        if self.max_pool:
            x = self.pooling(x)
        return x


class CNN(nn.Module):
    """
    Custom CNN architecture with three convolutional blocks.
    """
    def __init__(self, in_channels, hidden_units1, hidden_units2, output_shape):
        super().__init__()
        
        self.conv_block1 = ConvBlock(in_channels, hidden_units1, hidden_units1, max_pool=True)
        self.conv_block2 = ConvBlock(hidden_units1, hidden_units2, hidden_units2, max_pool=True)
        self.conv_block3 = ConvBlock(hidden_units2, hidden_units1, hidden_units1, max_pool=False)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units1 * 16 * 16, output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)
