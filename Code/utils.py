import torch
import joblib
from model import *
from preprocess.acf import *
from preprocess.gaussianize import *

# 생성자 네트워크를 초기화하는 함수
def initialize_generator(n_groups, G_type, noise_dim, n_samples, device):
    generators = {}
    for i in range(n_groups):
        if G_type == 'LSTM':
            generators[i] = LSTMGenerator(latent_dim=noise_dim, ts_dim=n_samples)
        elif G_type == 'TCN':
            generators[i] = TCNGenerator()
        generators[i] = nn.DataParallel(generators[i]).to(device)
    return generators

# 판별자 네트워크를 초기화하는 함수
def initialize_discriminator(n_groups, D_type, criterion, seq_len, n_samples, device):
    discriminators = {}
    for i in range(n_groups):
        if D_type == 'LSTM':
            if criterion == 'BCE':
                discriminators[i] = LSTMDiscriminator(ts_dim=n_samples, seq_len=seq_len)
        elif D_type == 'TCN':
            if criterion == 'BCE':
                discriminators[i] = TCNDiscriminator(seq_len=seq_len, input_dim=n_samples)
            elif criterion == 'WGAN-GP':
                discriminators[i] = TCNDiscriminator_GP(seq_len=seq_len, input_dim=n_samples)
        discriminators[i] = nn.DataParallel(discriminators[i]).to(device)
    return discriminators

# 중앙 판별자 네트워크를 초기화하는 함수
def initialize_central_discriminator(CD_type, criterion, n_groups, n_samples, seq_len, device):
    if CD_type == 'MLP':
        if criterion == 'BCE':
            central_discriminator = Central_Discriminator(n_samples=n_groups * n_samples, seq_len=seq_len, alpha=0.1)
            central_discriminator = central_discriminator.apply(initialize_weights)
        elif criterion == 'WGAN-GP':
            central_discriminator = Central_Discriminator_GP(n_samples=n_groups * n_samples, seq_len=seq_len, alpha=0.1)
            central_discriminator = central_discriminator.apply(initialize_weights)
    elif CD_type == 'TCN':
        if criterion == 'BCE':
            central_discriminator = TCNDiscriminator(seq_len=seq_len, input_dim=n_groups)
        elif criterion == 'WGAN-GP':
            central_discriminator = TCNDiscriminator_GP(seq_len=seq_len, input_dim=n_groups)
    central_discriminator = nn.DataParallel(central_discriminator).to(device)    
    return central_discriminator

### 입력 데이터 스케일링을 위한 함수 정의
# QuantGAN에서 제시한 데이터 스케일링 방법 사용 https://github.com/JamesSullivan/temporalCN
def scaling(data, n_groups):
    columns = []    
    for i in range(n_groups):        
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        gaussianize = Gaussianize()        
        log_returns = data[:, i].reshape(-1, 1)
        log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))        
        
        joblib.dump(standardScaler1, f'./Dataset/pickle/{i}_standardScaler1.pkl')
        joblib.dump(standardScaler2, f'./Dataset/pickle/{i}_standardScaler2.pkl')
        joblib.dump(gaussianize, f'./Dataset/pickle/{i}_gaussianize.pkl')    
        joblib.dump(log_returns, f'./Dataset/pickle/{i}_log_returns.pkl')                
        
        columns.append(log_returns_preprocessed.reshape(-1))
                        
    return np.array(columns).T

### WGAN-GP loss를 위한 gradient penalty 항 계산
# WGAN-GP 참조 https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    alpha = alpha.expand(real_samples.size())

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    fake = torch.autograd.Variable(torch.ones(d_interpolates.shape), requires_grad=False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

