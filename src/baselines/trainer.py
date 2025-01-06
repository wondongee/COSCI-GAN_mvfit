import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import os
import wandb

class GANTrainer:
    def __init__(self, config, train_loader, generators, discriminators, central_discriminator, device):
        self.config = config
        self.train_loader = train_loader
        self.generators = generators
        self.discriminators = discriminators
        self.central_discriminator = central_discriminator        
        self.device = device
        self.loss_function = nn.BCELoss()
        
        # Directory settings
        self.full_name = (
            f'{config.n_epochs}_{config.batch_size}_'
            f'gamma_{config.gamma}_Glr_{config.g_lr}_Dlr_{config.d_lr}_CDlr_{config.cd_lr}_'
            f'n_steps_{config.n_steps}_loss_{config.criterion}_layer1_split'
        )
        self.results_dir = f'./results/{self.full_name}/'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 각 네트워크의 optimizer 초기화
        self.optimizers_D = {i: torch.optim.Adam(discriminators[i].parameters(), lr=config.d_lr, betas=[0.5, 0.9]) for i in range(config.n_vars)}
        self.optimizers_G = {i: torch.optim.Adam(generators[i].parameters(), lr=config.g_lr, betas=[0.5, 0.9]) for i in range(config.n_vars)}
        self.optimizer_CD = torch.optim.Adam(central_discriminator.parameters(), lr=config.cd_lr, betas=[0.5, 0.9])
        
        # Initialize schedulers
        self.scheduler_G = {
            i: lr_scheduler.StepLR(self.optimizers_G[i], step_size=10, gamma=0.9)
            for i in range(config.n_vars)
        }
        self.scheduler_D = {
            i: lr_scheduler.StepLR(self.optimizers_D[i], step_size=10, gamma=0.9)
            for i in range(config.n_vars)
        }
        self.scheduler_CD = lr_scheduler.StepLR(self.optimizer_CD, step_size=10, gamma=0.9)

    def fit(self):        
        # Initialize wandb
        wandb.init(project='COSCI-GAN', config=self.config)

        for epoch in tqdm(range(self.config.n_epochs), desc="Epoch"):
            for n, real in enumerate(self.train_loader):
                real_batch = real[0].to(self.device)
                self.step(real_batch, n)

            # Update schedulers and save models every few epochs
            if (epoch + 1) % 5 == 0:
                for i in range(self.config.n_vars):
                    self.scheduler_G[i].step()
                    self.scheduler_D[i].step()
                self.scheduler_CD.step()

                # Save generator models
                for i in range(self.config.n_vars):
                    torch.save(self.generators[i].state_dict(), f'{self.results_dir}/Generator_{i}_{epoch+1}.pt')
                print(f"Saved Generator at epoch {epoch+1}")        
        print("Training complete. Models saved.")
        
        
    def step(self, real, n):
        batch_size = real.size(0)
        real = real.transpose(1, 2).to(self.device)

        shared_noise = torch.randn((batch_size, self.config.noise_dim, self.config.n_steps)).float().to(self.device)

        
        real_group = {i: real[:, i:(i+1)] for i in range(self.config.n_vars)}            
        fake_group = {i: self.generators[i](shared_noise) for i in range(self.config.n_vars)}                
        all_group = {i: torch.cat((real_group[i], fake_group[i])) for i in range(self.config.n_vars)}

        # Create labels
        real_labels = torch.ones((batch_size, 1)).to(self.device)
        fake_labels = torch.zeros((batch_size, 1)).to(self.device)
        all_labels = torch.cat((real_labels, fake_labels))

        ############################
        # (1) Training Discriminators
        ############################
        for i in range(self.config.n_vars):
            self.optimizers_D[i].zero_grad()
            outputs_D = self.discriminators[i](all_group[i].float())
            loss_D = self.loss_function(outputs_D, all_labels)
            loss_D.backward(retain_graph=True)
            self.optimizers_D[i].step()
            wandb.log({f"Discriminator Loss (Var {i})": loss_D.item()})

        ############################
        # (2) Training Central Discriminator
        ############################
        all_generated = torch.cat([fake_group[i] for i in range(self.config.n_vars)], dim=1)
        all_real = torch.cat([real_group[i] for i in range(self.config.n_vars)], dim=1)
        all_samples_central = torch.cat((all_generated, all_real))
        all_samples_labels = torch.cat(
            (torch.zeros((batch_size, 1)).to(self.device).float(), 
             torch.ones((batch_size, 1)).to(self.device).float())
        )

        self.optimizer_CD.zero_grad()
        output_CD = self.central_discriminator(all_samples_central.float())
        loss_CD = self.loss_function(output_CD, all_samples_labels)
        loss_CD.backward(retain_graph=True)
        self.optimizer_CD.step()
        wandb.log({"Central Discriminator Loss": loss_CD.item()})

        ############################
        # (3) Training Generators
        ############################
        
        for i in range(self.config.n_vars):
            self.optimizers_G[i].zero_grad()
                        
            fake_pred = self.discriminators[i](fake_group[i].float())
            gen_loss = self.loss_function(fake_pred, real_labels)  # Generator tries to make fake data as real

            # Central Discriminator loss
            fake_all_new = {j: self.generators[j](shared_noise) for j in range(self.config.n_vars)}
            fake_all = torch.cat([fake_all_new[j] for j in range(self.config.n_vars)], dim=1)            
            #fake_all = torch.cat([fake_group[j] for j in range(self.config.n_vars)], dim=1)
            
            all_new = torch.cat((fake_all, all_real))
            output_CD_new = self.central_discriminator(all_new.float())
            loss_CD_new = self.loss_function(output_CD_new, all_samples_labels)

            # Total Generator loss
            total_gen_loss = gen_loss - self.config.gamma * loss_CD_new
            total_gen_loss.backward(retain_graph=True)
            self.optimizers_G[i].step()

            wandb.log({f"Generator Loss (Var {i})": total_gen_loss.item()})