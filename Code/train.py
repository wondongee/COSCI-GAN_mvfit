import torch
from torch import nn
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle as pickle
import wandb
from model import *
from data_loader import *
from utils import *


def COSCIGAN(n_groups,
             dataset,
             num_epochs,
             batch_size,
             n_samples,
             criterion,
             G_type,
             D_type,
             CD_type,
             g_lr,
             d_lr,
             cd_lr,
             gamma,
             noise_dim,
             seq_len,
             ):
    # wandb에 실험 설정 저장 및 초기화 - 각 네트워크의 손실을 저장하기 위함    
    config = {
        "gen_lr" : g_lr, "dis_lr" : d_lr,
        "CD_lr" : cd_lr, "CD_type" : CD_type,
        "G_type" : G_type, "D_type" : D_type,
        "Gamma" : gamma
    }
    wandb.init(project='COSCI-GAN_Journal')
    wandb.config.update(config)
    
    full_name = f'{G_type}_{D_type}_{CD_type}_{num_epochs}_{batch_size}_gamma_{gamma}_Glr_{g_lr}_Dlr_{d_lr}_CDlr_{cd_lr}_seqlen_{seq_len}_loss_{criterion}_{wandb.run.name}'    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
           
    # 결과값 저장을 위한 디렉토리 생성
    if not os.path.isdir(f'./Results/'):
        os.mkdir(f'./Results/')
    if not os.path.isdir(f'./Results/{full_name}/'):
        os.mkdir(f'./Results/{full_name}/')
        
    # 데이터를 읽어와서 전처리 진행
    with open('./Dataset/'+dataset+'.csv', 'rb') as fh:
        df = pd.read_csv(fh)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    df = df.apply(pd.to_numeric).astype(float)    
    log_returns = np.diff(np.log(df), axis=0)
    log_returns_preprocessed = scaling(log_returns, n_groups)
    
    # 데이터를 sequence 단위로 나누어서 학습을 위한 dataloader로 변환
    dataset = dataloader(log_returns_preprocessed, seq_len)    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 각 네트워크 초기화
    generators = initialize_generator(n_groups, G_type, noise_dim, n_samples, device)
    discriminators = initialize_discriminator(n_groups, D_type, criterion, seq_len, n_samples, device)
    central_discriminator = initialize_central_discriminator(CD_type, criterion, n_groups, n_samples, seq_len, device)
    
    # 각 네트워크의 optimizer 및 손실함수 초기화
    optimizers_D = {}
    for i in range(n_groups):
        optimizers_D[i] = torch.optim.Adam(discriminators[i].parameters(), lr=d_lr, betas=[0.5, 0.9])
    optimizers_G = {}
    for i in range(n_groups):
        optimizers_G[i] = torch.optim.Adam(generators[i].parameters(), lr=g_lr, betas=[0.5, 0.9])
    optimizer_CD = torch.optim.Adam(central_discriminator.parameters(), lr=cd_lr, betas=[0.5, 0.9])
    
    if criterion == 'BCE':
        loss_function = nn.BCELoss()    
        
    # 학습 진행
    for epoch in tqdm(range(num_epochs)):
        log_Dict = {}
        for n, (real) in enumerate(train_loader):                                 
            # real - [batch_num, n_groups, seq_len]
            real = real.to(device)  
            batch_num = real.size(0)
            shared_noise = torch.randn((batch_num, noise_dim, seq_len)).float()                        
            
            # Split the data into groups
            real_group = {}
            for i in range(n_groups):
                real_group[i] = real[:, i*n_samples:(i+1)*n_samples]                                                                            
            fake_group = {}
            for i in range(n_groups):
                fake_group[i] = generators[i](shared_noise).float()                                                            
            all_group = {}                        
            for i in range(n_groups):
                all_group[i] = torch.cat((real_group[i], fake_group[i]))
                            
            # Create the labels
            fake_labels = torch.zeros((batch_num, 1)).to(device).float()
            real_labels = torch.ones((batch_num, 1)).to(device).float()
            all_labels = torch.cat((real_labels, fake_labels))

            ############################
            # (1) Training D network
            ###########################            
            outputs_D = {}
            loss_D = {}            
            for i in range(n_groups):
                optimizers_D[i].zero_grad()                                                
                if criterion == 'BCE':                    
                    outputs_D[i] = discriminators[i](all_group[i].float())
                    loss_D[i] = loss_function(outputs_D[i], all_labels)
                loss_D[i].backward(retain_graph=True)
                optimizers_D[i].step()

            ############################
            # (2) Training CD network
            ###########################            
            # concatenate the data to training CD network
            temp_generated = fake_group[0]
            for i in range(1,n_groups):
                temp_generated = torch.hstack((temp_generated, fake_group[i]))                    
            group_generated = temp_generated

            temp_real = real_group[0]
            for i in range(1,n_groups):
                temp_real = torch.hstack((temp_real, real_group[i]))                    
            group_real = temp_real     
            
            all_samples_central = torch.cat((group_generated, group_real))
            all_samples_labels = torch.cat(
                (torch.zeros((batch_num, 1)).to(device).float(), torch.ones((batch_num, 1)).to(device).float())
            )
            
            optimizer_CD.zero_grad()                        
            if criterion == 'BCE':
                output_CD = central_discriminator(all_samples_central.float())
                loss_CD = loss_function(output_CD, all_samples_labels)
            loss_CD.backward(retain_graph=True)
            optimizer_CD.step()
            
            ############################
            # (3) Training G network
            ###########################
            outputs_G = {}
            loss_G_local = {}
            loss_G = {}
            for i in range(n_groups):                
                
                ## Channel Discriminator로 부터 local loss 계산
                optimizers_G[i].zero_grad()
                outputs_G[i] = discriminators[i](fake_group[i])                            
                if criterion == 'BCE':
                    loss_G_local[i] = loss_function(outputs_G[i], real_labels)
                
                ## Central Discriminator로 부터 central loss 계산
                all_new = {}
                fake_new = {}
                output_CD_new = {}
                loss_CD_new = {}                
                for j in range(n_groups):
                    fake_new[j] = generators[j](shared_noise)
                    if i == j:
                        fake_new[j] = fake_new[j].float()
                    else:
                        fake_new[j] = fake_new[j].detach().float()                                
                fake_temp = fake_new[0]
                for j in range(1, n_groups):
                    fake_temp = torch.hstack((fake_temp, fake_new[j]))
                fake_all = fake_temp                
                all_new[i] = torch.cat((fake_all, group_real))   
                
                ## 생성자의 total loss 계산 = local loss - gamma * central loss                
                if criterion == 'BCE':
                    output_CD_new[i] = central_discriminator(all_new[i].float()) 
                    loss_CD_new[i] = loss_function(output_CD_new[i], all_samples_labels)                    
                    loss_G[i] = loss_G_local[i] - gamma * loss_CD_new[i]                                    
                loss_G[i].backward(retain_graph=True)
                optimizers_G[i].step()                
                
                ### 생성자의 손실값 저장
                log_Dict[f'loss_G_{i}'] = loss_G[i].cpu()            
                wandb.log({f"Generator_loss_{i}": loss_G[i]})
                    
        ### 판별자 및 중앙 판별자의 손실값 저장
        for i in range(n_groups):
            log_Dict[f'loss_D_{i}'] = loss_D[i].cpu()
            wandb.log({f"Discriminator_loss_{i}": loss_D[i]})        
        log_Dict['loss_CD'] = loss_CD.cpu()
        wandb.log({"Central_Discriminator_loss": loss_CD})                                          
        
        ### 모델 결과값 저장 - 생성자의 State_dict
        if epoch % 5 == 0:
            for i in range(n_groups):
                torch.save(generators[i].state_dict(), f'./Results/{full_name}/Generator_{i}_{epoch}.pt')
                print(f"Save the Generator_{epoch}")                              