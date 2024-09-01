import torch
import torch.nn as nn
import torch.nn.functional as F

###################### CNN ###############################
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
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.resblock1 = self._make_res_block(8, 8, 1)
        
        self.layer2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.resblock2 = self._make_res_block(16, 16, 1)
        
        self.layer3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.resblock3 = self._make_res_block(32, 32, 1)
        
        self.layer4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.resblock4 = self._make_res_block(64, 64, 1)
        
        self.layer5 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.feature_dim = None
        
    def _make_res_block(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ResNetBlock(in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _forward_conv_layers(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.layer2(x)
        x = self.resblock2(x)
        x = self.layer3(x)
        x = self.resblock3(x)
        x = self.layer4(x)
        x = self.resblock4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    
    def _initialize_fc(self, input_shape, device):
        # Forward pass through conv layers to calculate the size of the flatten layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)
            dummy_output = self._forward_conv_layers(dummy_input)
            self.feature_dim = dummy_output.numel()
        self.fc = nn.Linear(self.feature_dim, 64)

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.flatten(x)
        if self.feature_dim is None:
            self.feature_dim = x.size(1)
        x = self.fc(x)   
        return x

###################### CNN LSTM ###################################
class CNN_LSTM(nn.Module):
    def __init__(self, cnn, input_shape, seq_len):
        super(CNN_LSTM, self).__init__()
        self.cnn = cnn
        self.hidden_size = 128
        self.lstm = nn.LSTM(64, seq_len*128,num_layers=1, batch_first=True)
        self.fc = nn.Linear(seq_len*128, 2)
        self.seq_len = seq_len
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn.to(device)
        self.cnn._initialize_fc(input_shape, device)
        self.layer_norm = nn.LayerNorm(64)



    def forward(self, x):
        device = x.device
        batch_size, seq_len, channels, height, width = x.size()
        cnn_out = []
        for i in range(seq_len):
            out = self.cnn(x[:, i, :, :, :].to(device))
            out = self.layer_norm(out)
            cnn_out.append(out)
        r_in = torch.stack(cnn_out, dim=1)  # Reshape for LSTM input
        h0 = torch.zeros(1, batch_size, seq_len*self.hidden_size).to(x.device)  # Initial hidden state
        c0 = torch.zeros(1, batch_size, seq_len*self.hidden_size).to(x.device)  # Initial cell state
        r_out, _ = self.lstm(r_in, (h0, c0))
        out = self.fc(r_out)  # Get the last time step output
        return out    

def AstroNet():
    input_shape=(1, 256, 256)
    seq_len =3
    cnn = CustomCNN()
    model=CNN_LSTM(cnn, input_shape, seq_len)
    return model


############################# CNN RNN ###########################################


class CNN_RNN(nn.Module):
    def __init__(self, input_shape, seq_len):
        super(CNN_RNN, self).__init__()
        self.cnn = CustomCNN()
        self.seq_len = seq_len

        # Initialize CNN to get feature dimension
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn.to(device)
        self.cnn._initialize_fc(input_shape, device)
        self.layer_norm = nn.LayerNorm(64)
        # Initialize RNN and Fully Connected layers
        self.rnn = nn.RNN(64, seq_len*256, num_layers=2,dropout=0.5, batch_first=True)
        self.fc = nn.Linear(seq_len*256, 2) # Output two continuous values

    def forward(self, x):
        device = x.device
        batch_size, seq_len, c, h, w = x.size()
        
        cnn_out = []
        for i in range(seq_len):
            out = self.cnn(x[:, i, :, :, :].to(device))
            out = self.layer_norm(out)
            cnn_out.append(out)
        
        cnn_out = torch.stack(cnn_out, dim=1)
        rnn_out, _ = self.rnn(cnn_out)
        
        # Only take the output from the last RNN cell
        out = self.fc(rnn_out)

        return out

def RNN():
    # Hyperparameters
    seq_len=3
    input_shape=(1, 256, 256)
    model = CNN_RNN(input_shape, seq_len)  # Example input shape and sequence length
    return model



