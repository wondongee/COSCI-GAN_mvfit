import joblib
from model import *
from preprocess.gaussianize import *
from sklearn.preprocessing import StandardScaler
import random

# 생성자 네트워크를 초기화하는 함수
def initialize_generator(n_groups, G_type, noise_dim, n_samples, device):
    generators = {}
    for i in range(n_groups):
        if G_type == 'LSTM':
            generators[i] = LSTMGenerator(latent_dim=noise_dim, ts_dim=n_samples)        
        generators[i] = nn.DataParallel(generators[i]).to(device)
    return generators

# 판별자 네트워크를 초기화하는 함수
def initialize_discriminator(n_groups, D_type, criterion, seq_len, n_samples, device):
    discriminators = {}
    for i in range(n_groups):
        if D_type == 'LSTM':
            if criterion == 'BCE':
                discriminators[i] = LSTMDiscriminator(ts_dim=n_samples, seq_len=seq_len)
        discriminators[i] = nn.DataParallel(discriminators[i]).to(device)
    return discriminators

# 중앙 판별자 네트워크를 초기화하는 함수
def initialize_central_discriminator(CD_type, criterion, n_groups, n_samples, seq_len, device):
    if CD_type == 'MLP':
        if criterion == 'BCE':
            central_discriminator = Central_Discriminator(n_samples=n_groups * n_samples, seq_len=seq_len, alpha=0.1)
            central_discriminator = central_discriminator.apply(initialize_weights)
    central_discriminator = nn.DataParallel(central_discriminator).to(device)    
    return central_discriminator

### 입력 데이터 스케일링을 위한 함수 정의
# QuantGAN에서 제시한 데이터 스케일링 방법 사용 https://github.com/JamesSullivan/temporalCN
def scaling(data, n_groups):
    columns = []    
    for i in range(n_groups):        
        standardScaler1 = StandardScaler()
        #standardScaler2 = StandardScaler()
        #gaussianize = Gaussianize()        
        log_returns = data[:, i].reshape(-1, 1)
        log_returns_preprocessed = standardScaler1.fit_transform(log_returns)        
        
        joblib.dump(standardScaler1, f'./Dataset/pickle/{i}_standardScaler1.pkl')
        #joblib.dump(standardScaler2, f'./Dataset/pickle/{i}_standardScaler2.pkl')
        #joblib.dump(gaussianize, f'./Dataset/pickle/{i}_gaussianize.pkl')    
        joblib.dump(log_returns, f'./Dataset/pickle/{i}_log_returns.pkl')                
        
        columns.append(log_returns_preprocessed.reshape(-1))
                        
    return np.array(columns).T


def inverse_process(y, asset_idx):
    standardScaler1 = joblib.load(f'./Dataset/pickle/{asset_idx}_standardScaler1.pkl')
    #standardScaler2 = joblib.load(f'./Dataset/pickle/{asset_idx}_standardScaler2.pkl')
    #gaussianize = joblib.load(f'./Dataset/pickle/{asset_idx}_gaussianize.pkl')
    log_returns = joblib.load(f'./Dataset/pickle/{asset_idx}_log_returns.pkl')
     
    #y = (y - y.mean(axis=0)) / y.std(axis=0)
    y = standardScaler1.inverse_transform(y)
    #y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
    #y = standardScaler1.inverse_transform(y)
    return y, log_returns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # Ensure deterministic behavior on GPU (if this is desired)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False