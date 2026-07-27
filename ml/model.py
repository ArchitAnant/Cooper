import torch
import torch.nn as nn

SAMPLE_RATE = 16000
CLIP_DURATION_SAMPLES = 16000  # 1 second
NUM_MFCC = 10
TARGET_WORDS = ['yes', 'no', 'zero', 'one', 'two', 'three', 'left', 'right', 'up', 'down']
# We append 'unknown' and 'silence' to the end
CLASSES = TARGET_WORDS + ['unknown', 'silence']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

NUM_CLASSES = len(CLASSES)

class DSCNNBlock(nn.Module):
    """A standard Depthwise Separable Convolution block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Depthwise Convolution (groups = in_channels)
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 2. Pointwise Convolution (1x1 standard conv)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class MicroSpeechDSCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Initial standard convolution to extract base features
        # Input shape: [Batch, 1, 10 (freq), 49 (time)]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stack 5 DS-CNN Layers (Highly parameter efficient, fits in 12KB-64KB SRAM)
        self.ds_conv1 = DSCNNBlock(16, 32)
        self.ds_conv2 = DSCNNBlock(32, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.ds_conv3 = DSCNNBlock(32, 32)
        self.ds_conv4 = DSCNNBlock(32, 16)
        
        # The Final block outputs EXACTLY 8 channels (as requested by the document)
        self.ds_conv5 = DSCNNBlock(16, 8)
        
        # Global Average Pooling (Solves the "Fatal Flaw" - ensures Temporal/Shift Invariance)
        # Squeezes spatial dimensions (H, W) down to 1x1.
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final Dense Layer
        # Since GAP outputs 8 values, and we have 4 classes: 8 * 4 = 32 weights
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.pool2(x)
        
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        x = self.ds_conv5(x)
        
        x = self.gap(x)
        
        # Flatten the (Batch, 8, 1, 1) output to (Batch, 8)
        x = torch.flatten(x, 1) 
        
        x = self.classifier(x)
        return x