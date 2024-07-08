from train import COSCIGAN
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# COSCI-GAN 디렉토리로 이동
path = '/workspace/COSCI-GAN_Journal'
try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory {0} does not exist".format(path))
    
# config 파일에서 설정 읽기
with open('./configs/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# 모델 학습을 위한 함수 호출
COSCIGAN(n_groups=config['Ngroups'],
         dataset=config['dataset'],
         num_epochs=config['nepochs'],
         batch_size=config['batch_size'],
         n_samples=config['nsamples'],         
         criterion=config['criterion'],
         G_type=config['G_type'],
         D_type=config['D_type'],
         CD_type=config['CD_type'],
         g_lr=config['glr'],
         d_lr=config['dlr'],
         cd_lr=config['cdlr'],
         gamma=config['gamma'],
         noise_dim=config['noise_dim'],
         seq_len=config['seq_len']         
        )