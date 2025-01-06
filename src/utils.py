from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import ml_collections
import yaml
import random

def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def count_parameters(model: torch.nn.Module) -> int:
    """

    Args:
        model (torch.nn.Module): input models
    Returns:
        int: number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


"""
class BaseAugmentation:
    pass

    def apply(self, *args: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('Needs to be implemented by child.')


@dataclass
class AddTime(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)
"""


def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(
        dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


def get_experiment_dir(config):
    if config.model_type == 'VAE':
        exp_dir = './numerical_results/{dataset}/algo_{gan}_Model_{model}_n_lag_{n_lags}_{seed}'.format(
            dataset=config.dataset, gan=config.algo, model=config.model, n_lags=config.n_lags, seed=config.seed)
    else:
        exp_dir = './numerical_results/{dataset}/algo_{gan}_G_{generator}_D_{discriminator}_includeD_{include_D}_n_lag_{n_lags}_{seed}'.format(
            dataset=config.dataset, gan=config.algo, generator=config.generator,
            discriminator=config.discriminator, include_D=config.include_D, n_lags=config.n_lags, seed=config.seed)
    os.makedirs(exp_dir, exist_ok=True)
    if config.train and os.path.exists(exp_dir):
        print("WARNING! The model exists in directory and will be overwritten")
    config.exp_dir = exp_dir


def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def loader_to_cond_tensor(dl, config):
    tensor = []
    for _, y in dl:
        tensor.append(y)

    return one_hot(torch.cat(tensor), config.num_classes).unsqueeze(1).repeat(1, config.n_lags, 1)

def combine_dls(dls):
    return torch.cat([loader_to_tensor(dl) for dl in dls])


def fake_loader(generator, x_past, n_lags, batch_size, **kwargs):
    """
    Helper function that transforms the generated data into dataloader, adapted from different generative models
    Parameters
    ----------
    generator: nn.module, trained generative model
    x_past: torch.tensor, real past path
    num_samples: int,  number of paths to be generated
    n_lags: int, the length of path to be generated
    batch_size: int, batch size for dataloader
    kwargs

    Returns
    Dataload of generated data
    -------

    """
    with torch.no_grad():
        fake_data_future = generator(n_lags, x_past)
        fake_data = torch.cat([x_past, fake_data_future], dim=1)
    return DataLoader(TensorDataset(fake_data), batch_size=batch_size)

def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor
def deterministic_NeuralSort(s, tau):
    """
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar. default: 0.01
    """
    n = s.size()[1]
    one = torch.ones((n, 1)).type(Tensor).to("cuda")
    A_s = torch.abs(s - s.permute(0, 2, 1)).to("cuda")
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1).to("cuda")))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).type(Tensor)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat
        
from sklearn.preprocessing import StandardScaler
import numpy as np

""" Scale the log returns using StandardScaler only """
def scaling(data):
    """
    각 피처별로 StandardScaler를 적용하여 데이터를 스케일링합니다.

    Parameters:
        data (np.ndarray): 스케일링할 데이터, shape=(samples, features)

    Returns:
        scaled_data (np.ndarray): 스케일링된 데이터, shape=(samples, features)
        scalers (list of StandardScaler): 각 피처별로 학습된 StandardScaler 객체 리스트
    """
    scalers = [] 
    scaled_data = []

    for i in range(data.shape[1]):  # 피처별로 반복
        scaler = StandardScaler()

        # 현재 피처의 데이터 추출 및 스케일링
        feature_data = data[:, i].reshape(-1, 1)  # shape=(samples, 1)
        feature_scaled = scaler.fit_transform(feature_data)

        # 스케일러 및 스케일링된 데이터 저장
        scalers.append(scaler)
        scaled_data.append(feature_scaled.flatten())  # shape=(samples,)

    # 스케일링된 피처들을 다시 합쳐서 (samples, features) 형태로 변환
    scaled_data = np.array(scaled_data).T  # shape=(samples, features)
    return scaled_data, scalers


""" Inverse the scaling process for all features using StandardScaler only """
def inverse_scaling(y, scalers):
    """
    스케일링된 데이터를 원래대로 복원합니다.

    Parameters:
        y (torch.Tensor): 스케일링된 데이터, shape=(batch_size, features, seq_len)
        scalers (list of StandardScaler): 각 피처별로 학습된 StandardScaler 객체 리스트

    Returns:
        y_original (np.ndarray): 복원된 원본 데이터, shape=(batch_size, features, seq_len)
    """
    y = y.cpu().detach().numpy()  # torch.Tensor를 NumPy 배열로 변환
    y_original = np.zeros_like(y)  # 원본 데이터를 저장할 배열 초기화

    for idx in range(y.shape[1]):
        scaler = scalers[idx]
        
        y_feature = y[:, idx, :]  # shape=(batch_size, seq_len)

        # 각 피처별로 스케일링을 원상복구
        # reshape하여 StandardScaler가 요구하는 2D 형태로 변환
        y_feature_reshaped = y_feature.reshape(-1, 1)  # shape=(batch_size * seq_len, 1)
        y_feature_original = scaler.inverse_transform(y_feature_reshaped)
        
        # 다시 원래의 shape으로 복원
        y_original[:, idx, :] = y_feature_original.reshape(y_feature.shape)

    return y_original
