import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        return out
    
class CustomCNN(nn.Module):
    def __init__(self, input_shape=(1, 256, 256)):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Reduced channels
        self.resblock1 = self._make_res_block(8, 8, 1)  # Fewer blocks
        
        self.layer2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # Reduced channels
        self.resblock2 = self._make_res_block(16, 16, 1)  # Fewer blocks
        
        self.layer3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Reduced channels
        self.resblock3 = self._make_res_block(32, 32, 1)  # Fewer blocks
        
        self.layer4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Reduced channels
        self.resblock4 = self._make_res_block(64, 64, 1)  # Fewer blocks
        
        self.layer5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Reduced channels
        self.layer6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Reduced channels
        
        self.flatten = nn.Flatten()
        self._initialize_fc(input_shape)
        
    def _initialize_fc(self, input_shape):
        # Forward pass through conv layers to calculate the size of the flatten layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self._forward_conv_layers(dummy_input)
            flattened_size = dummy_output.numel()
        self.feature_dim = flattened_size  # Save the feature dimension for the LSTM
        self.fc = nn.Linear(flattened_size, 2)
        

    def _make_res_block(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResNetBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_conv_layers(self, x):
        x = F.leaky_relu(self.layer1(x), negative_slope=0.01, inplace=True)
        x = self.resblock1(x)
        
        x = F.leaky_relu(self.layer2(x), negative_slope=0.01, inplace=True)
        x = self.resblock2(x)
        
        x = F.leaky_relu(self.layer3(x), negative_slope=0.01, inplace=True)
        x = self.resblock3(x)
        
        x = F.leaky_relu(self.layer4(x), negative_slope=0.01, inplace=True)
        x = self.resblock4(x)
        
        x = F.leaky_relu(self.layer5(x), negative_slope=0.01, inplace=True)
        x = F.leaky_relu(self.layer6(x), negative_slope=0.01, inplace=True)
        
        return x

    def extract_features(self, x):
        x = self._forward_conv_layers(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        output = self.fc(features)
        return output


    

class BrightnessCenterCNN(nn.Module):
    def __init__(self):
        super(BrightnessCenterCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)        
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        
        # Calculate the input size for the fully connected layer
        # Assuming the input image size is 256x256
        conv_output_size = 256 // (2 ** 4)  # 4 pooling layers each with stride 2
        self.fc1 = nn.Linear(256 * conv_output_size * conv_output_size, 2)  # Output size is 2
        
        self.dropout = nn.Dropout(0.5)  # Dropout with 50% probability
        
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))
        
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        
        x = self.flatten(x)
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc1(x)
        
        return x

def CNN(input_shape=(1, 256, 256)):
    model = CustomCNN(input_shape)

    return model

def ResNet():
    model = BrightnessCenterCNN()
    return model


