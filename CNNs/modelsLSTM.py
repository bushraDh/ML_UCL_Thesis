import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNN_LSTM(nn.Module):
#     def __init__(self, cnn_model, sequence_length, hidden_dim=64, num_layers=1, dropout=0.2):
#         super(CNN_LSTM, self).__init__()
#         self.cnn_model = cnn_model
#         self.lstm = nn.LSTM(input_size=cnn_model.feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_dim, 2)  # Output (x, y) coordinates

#     def forward(self, x):
#         batch_size, seq_length, channels, height, width = x.size()
#         x = x.view(batch_size * seq_length, channels, height, width)
#         features = self.cnn_model.extract_features(x)
#         features = features.view(batch_size, seq_length, -1)
#         lstm_out, (hn, cn) = self.lstm(features)
#         output = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM
#         return output

######################CNN LSTM###################################
class CNN_FE(nn.Module):
    def __init__(self):
        super(CNN_FE, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # Calculate the size after the second pooling layer
        # Input size: 1x256x256
        # After conv1 and pool: 8x128x128
        # After conv2 and pool: 16x64x64
        self.fc = nn.Linear(16 * 64 * 64, 128)  # Adjusted for 16 * 64 * 64 output size

    def forward(self, x):
        x = self.pool(self.leaky_relu(self.conv1(x)))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = x.view(-1, 16 * 64 * 64)  # Flatten the tensor
        x =self.fc(x)
        return x
    
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

def CNNLSTM():
    input_shape=(1, 256, 256)
    seq_len =3
    cnn = CustomCNN().cuda()
    model=CNN_LSTM(cnn, input_shape, seq_len)
    return model

################################## CONV LSTM ##################################################
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # Handle tuple for padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Ensure that hidden_dim and kernel_size are lists with length num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        if not len(hidden_dim) == len(kernel_size) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class ConvLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim, image_size):
        super(ConvLSTMModel, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  kernel_size=kernel_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)
        self.conv1 = nn.Conv2d(hidden_dim[-1], 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * (image_size[0] // 2) * (image_size[1] // 2), output_dim)

    def forward(self, x):
        layer_output_list, _ = self.conv_lstm(x)
        x = layer_output_list[0][:, -1, :, :, :]  # Take the output of the last timestep
        x = self.pool(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def Conv_LSTM():
    input_dim = 1  # Grayscale images
    hidden_dim = [32]  # Single layer with 32 hidden units
    kernel_size = (3, 3)  # Kernel size for ConvLSTM
    num_layers = 1  # Number of ConvLSTM layers
    output_dim = 2  # Predicting (x, y) coordinates
    image_size = (256, 256)  # Image dimensions after resizing

    model = ConvLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                          num_layers=num_layers, output_dim=output_dim, image_size=image_size)
    return model

############################# RNN ###########################################

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

def RNN1():
    # Hyperparameters
    model = CNN_RNN(input_shape=(1, 256, 256), seq_len=3)  # Example input shape and sequence length
    return model



