import torch
from torch import nn
from torch.nn.utils import weight_norm

def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

# Central Discriminator with MLP for BCE loss
class Central_Discriminator(nn.Module):
    def __init__(self, n_samples, seq_len, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_samples*seq_len, 256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output

# Central Discriminator with MLP for WGAN-GP loss 
class Central_Discriminator_GP(nn.Module):
    def __init__(self, n_samples, seq_len, alpha):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_samples*seq_len, 256),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output

# Generator with LSTM
class LSTMGenerator(nn.Module):
    """Generator with LSTM"""
    def __init__(self, latent_dim, ts_dim, hidden_dim=64, num_layers=1):
        super(LSTMGenerator, self).__init__()

        self.ts_dim = ts_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, ts_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.linear(out)
        out = out.permute(0, 2, 1)
        return out
    
# Discriminator with LSTM
class LSTMDiscriminator(nn.Module):
    def __init__(self, ts_dim, seq_len, hidden_dim=64, num_layers=1):
        super(LSTMDiscriminator, self).__init__()        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(ts_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())
    def forward(self, x):
        x = x.permute(0, 2, 1)            
        out, _ = self.lstm(x)
        out = self.linear(out).squeeze()
        out = self.to_prob(out)
        return out

# Generator with TCN 
class TCNGenerator(nn.Module):    
    def __init__(self):
        super(TCNGenerator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(3, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x

# Discriminator with TCN for BCE loss
class TCNDiscriminator(nn.Module):    
    def __init__(self, seq_len, input_dim, conv_dropout=0.05):
        super(TCNDiscriminator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(input_dim, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)                
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).reshape(-1, 1)

# Discriminator with TCN for WGAN-GP loss
class TCNDiscriminator_GP(nn.Module):
    def __init__(self, seq_len, input_dim, conv_dropout=0.05):
        super(TCNDiscriminator_GP, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(input_dim, 80, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(80, 80, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(80, 1, kernel_size=1, dilation=1)                
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1))

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()
    
# Temporal Convolutional Block
class TemporalBlock(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.5)
        self.conv2.weight.data.normal_(0, 0.5)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.5)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out, self.relu(out + res)
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()