import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance, ks_2samp, spearmanr, kendalltau
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon
from scipy.stats import skew, kurtosis


def rolling_window(time_series, window_length):
    """
    Compute rolling windows from a time series.

    Args:
        time_series (2D array): Input time series of shape (timesteps, samples).
        window_length (int): Length of the rolling window.        

    Returns:
        3D array: Array of rolling windows of shape (window_count, window_length, samples).
    """
    # Calculate the number of rolling windows
    window_count = time_series.shape[0] - window_length + 1
    
    # Create an array to store the rolling windows
    rolling_windows = np.zeros((window_count, window_length, time_series.shape[1]))
    
    # Use np.roll to efficiently compute rolling windows
    for i in range(window_count):
        rolling_windows[i] = time_series[i:i+window_length]
    
    return np.transpose(rolling_windows, axes=(1, 0, 2))

def rolling_window_seqlen(arr, window_size):
    result = []    
    for i in range(0, len(arr) - window_size + 1):
        if window_size + 1 + i > len(arr):
            break
        result.append(arr[i:i+window_size])
    return result


def calculate_correlation(df_merge, asset_length):

    corr_matrices = []

    for index, row in df_merge.iterrows():
        df1 = row[0:asset_length].values
        df2 = row[asset_length:asset_length*2].values
        df3 = row[asset_length*2:asset_length*3].values
        df4 = row[asset_length*3:asset_length*4].values
        df5 = row[asset_length*4:asset_length*5].values
        df_final = pd.DataFrame({'Asset1': df1, 'Asset2': df2, 'Asset3': df3, 'Asset4': df4, 'Asset5': df5})
        corr_matrix = df_final.corr()
        corr_matrices.append(corr_matrix.values)
    
    corr_matrices = np.array(corr_matrices)
    avg_corr_matrix = np.mean(corr_matrices, axis=0)
    std_corr_matrix = np.std(corr_matrices, axis=0)

    avg_corr = pd.DataFrame(avg_corr_matrix, columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'], index=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])
    std_corr = pd.DataFrame(std_corr_matrix, columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'], index=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])

    print('\nMean of synthetic data correlation')
    print(avg_corr)
    print()
    print('Standard Deviation of synthetic data correlation')
    print(std_corr)

    return corr_matrices, avg_corr, std_corr

def calculate_distribution_scores(real, fake, num_G, windows):
    scores = {
        'EMD': np.zeros((num_G, len(windows))),
        'KL': np.zeros((num_G, len(windows))),
        'JS': np.zeros((num_G, len(windows))),
        'KS': np.zeros((num_G, len(windows)))
    }

    for i in range(num_G):
        for j in range(len(windows)):
            real_dist = rolling_window(real[i], windows[j]).sum(axis=0).ravel()
            fake_dist = rolling_window(fake[i].T, windows[j]).sum(axis=0).ravel()
            
            np.random.shuffle(real_dist)
            np.random.shuffle(fake_dist)
            
            # Calculate EMD
            scores['EMD'][i, j] = wasserstein_distance(real_dist, fake_dist)
            
            # Calculate KS Statistic
            scores['KS'][i, j], _ = ks_2samp(real_dist, fake_dist)                                    
            
            # Create histograms to estimate the probability distributions
            real_hist, bin_edges = np.histogram(real_dist, bins=100, density=True)
            fake_hist, _ = np.histogram(fake_dist, bins=bin_edges, density=True)
            
            # Normalize the histograms to get probability distributions
            real_prob = real_hist / np.sum(real_hist)
            fake_prob = fake_hist / np.sum(fake_hist)
            
            # Calculate KL Divergence
            kl_divergence = entropy(real_prob + 1e-10, fake_prob + 1e-10)
            scores['KL'][i, j] = kl_divergence

            # Calculate JS Divergence
            js_divergence = jensenshannon(real_prob + 1e-10, fake_prob + 1e-10)
            scores['JS'][i, j] = js_divergence
         
            
    df_scores = {}
    for metric, data in scores.items():
        data = np.round(data, decimals=4)
        df_scores[metric] = pd.DataFrame(data.T, index=windows, columns=[f'{metric} {i}' for i in range(num_G)])
        
    emd_avg = np.mean(scores['EMD'], axis=0)
    df_scores['EMD']['EMD_avg'] = np.round(emd_avg, decimals=4)
    
    return df_scores

def calculate_mae(A, B): 
    A_values = A.values
    B_values = B.values        
    mae = np.mean(np.abs(A_values - B_values))
    return mae

def calculate_frobenius(A, B):
    A_values = A.values
    B_values = B.values
    frobenius_norm = np.linalg.norm(A_values - B_values, 'fro')
    return frobenius_norm

def calculate_spearman(A, B):
    A_values = A.values
    B_values = B.values
    rho, _ = spearmanr(A_values.flatten(), B_values.flatten())
    return rho

def calculate_kendall(A, B):
    A_values = A.values
    B_values = B.values
    tau, _ = kendalltau(A_values.flatten(), B_values.flatten())
    return tau

def calculate_mae_mean(A, B_array):
    mae_list = []
    A_values = A.values
    for row in B_array:
        B = row.reshape(A_values.shape)
        mae = np.mean(np.abs(A_values - B))
        mae_list.append(mae)        

    mae_mean = np.mean(mae_list)
    mae_std = np.std(mae_list)
    return mae_mean, mae_std



def plot_acf_comparison(real_list, fake_list, num_G, lags=50):    
    data_types = ['Identity', 'Absolute', 'Squared']
    data_transforms = [lambda x: x, np.abs, np.square]  # Identity 변환 추가
    titles = ['Identity log returns', 'Absolute log returns', 'Squared log returns']

    for i in range(num_G):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))  # 3개의 서브플롯 생성

        for ax, data_type, transform, title in zip(axs, data_types, data_transforms, titles):
            # 데이터 변환 및 ACF 계산
            transformed_real = transform(real_list[i])
            transformed_fake = transform(fake_list[i])
            
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])
            
            mean_real = np.mean(acf_real, axis=0)
            std_real = np.std(acf_real, axis=0)
            mean_fake = np.mean(acf_fake, axis=0)
            std_fake = np.std(acf_fake, axis=0)
            
            # 평균 ACF와 표준편차 기반 범위 표시
            ax.plot(mean_real, label=f'{data_type} ACF Real - Mean', color='tab:blue')
            ax.fill_between(range(lags+1), mean_real - 0.5*std_real, mean_real + 0.5*std_real, color='tab:blue', alpha=0.2,
                           label=f'{data_type} ACF Real - 1/2 Std Dev')        
            ax.plot(mean_fake, label=f'{data_type} ACF COMFI-GAN - Mean', color='tab:orange')
            ax.fill_between(range(lags+1), mean_fake - 0.5*std_fake, mean_fake + 0.5*std_fake, color='tab:orange', alpha=0.2,
                            label=f'{data_type} ACF COMFI-GAN - 1/2 Std Dev')

            # 플롯 커스터마이징
            ax.set_ylim(-0.10, 0.3)
            ax.set_title(title)
            ax.grid(True)
            ax.axhline(y=0, color='k')
            ax.axvline(x=0, color='k')
            ax.set_xlabel('Lag (number of days)')
            ax.legend()

        plt.show()
        
def calculate_acf_score(real_list, fake_list, lags=50):
    data_transforms = [lambda x: x, np.abs, np.square]  # Identity 변환 추가
    titles = ['Identity log returns', 'Absolute log returns', 'Squared log returns']
    acf_scores = {}
    
    n_groups = len(real_list)
    
    for i in range(n_groups):
        group_scores = {}
        
        for transform, title in zip(data_transforms, titles):
            transformed_real = transform(real_list[i])
            transformed_fake = transform(fake_list[i])
            
            acf_real = np.array([acf(ts, nlags=lags) for ts in transformed_real])
            acf_fake = np.array([acf(ts, nlags=lags) for ts in transformed_fake])
            
            mean_real = np.mean(acf_real, axis=0)
            mean_fake = np.mean(acf_fake, axis=0)
    
            group_scores[title] = np.round(np.linalg.norm(mean_real - mean_fake), decimals=4)
        
        acf_scores[f'Group {i+1}'] = group_scores
    
    return acf_scores

def calculate_skew_kurt_diff(fake_list, real_list):
    results = []
    
    for asset_idx in range(len(real_list)):
        fake_data = fake_list[asset_idx]
        real_data = real_list[asset_idx]

        # 각 샘플에 대해 skewness와 kurtosis를 계산하고 평균을 구합니다.
        fake_skewness = np.mean([skew(sample) for sample in fake_data])
        fake_kurtosis = np.mean([kurtosis(sample, fisher=False) for sample in fake_data])

        real_skewness = np.mean([skew(sample) for sample in real_data])
        real_kurtosis = np.mean([kurtosis(sample, fisher=False) for sample in real_data])

        # skewness와 kurtosis의 차이 계산
        skewness_diff = np.abs(real_skewness - fake_skewness)
        kurtosis_diff = np.abs(real_kurtosis - fake_kurtosis)

        # 소수점 자릿수 제한
        skewness_diff = round(skewness_diff, 4)
        kurtosis_diff = round(kurtosis_diff, 4)

        results.append((skewness_diff, kurtosis_diff))

        print(f'Asset {asset_idx + 1} Skewness Difference: {skewness_diff}')
        print(f'Asset {asset_idx + 1} Kurtosis Difference: {kurtosis_diff}')
    
    return results


def calculate_leverage_effect(real_list, fake_list, lags=50):
    def leverage_effect(ts, tau):
        rt = ts[:-tau]  # rt: 시계열에서 뒤에서 tau만큼 제외한 것
        rt_squared = (ts[tau:]**2)  # rt_tau_squared: 시계열에서 앞에서 tau만큼 제외하고 제곱한 것
        return np.corrcoef(rt, rt_squared)[0, 1]

    n_groups = len(real_list)
    leverage_scores = {}

    for i in range(n_groups):
        real_data = real_list[i]
        fake_data = fake_list[i]

        real_leverage_effects = np.array([[leverage_effect(ts, tau) for tau in range(1, lags+1)] for ts in real_data])
        fake_leverage_effects = np.array([[leverage_effect(ts, tau) for tau in range(1, lags+1)] for ts in fake_data])

        mean_real_leverage = np.mean(real_leverage_effects, axis=0)
        mean_fake_leverage = np.mean(fake_leverage_effects, axis=0)
        
        leverage_score = np.linalg.norm(mean_real_leverage - mean_fake_leverage)
        
        leverage_scores[f'Group {i+1}'] = np.round(leverage_score, decimals=4)
        
        # Print the result for the current group
        print(f'Group {i+1} leverage score: {leverage_scores[f"Group {i+1}"]}')

    return leverage_scores
