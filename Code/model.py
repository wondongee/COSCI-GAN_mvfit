import torch
from torch import nn

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
