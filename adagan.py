import os
from os.path import join as ospj
import tqdm
import re

from glob import glob

import torch
from torch.autograd import Variable
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import networks.dcgan as dcgan

import matplotlib.pyplot as plt
import torchvision.utils as vutils

class ADAGAN():
    def __init__(self, args):
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.args = args
        self.model_name = args.arch
        
        self.image_size = args.image_size
        
        self.advloss = args.advloss
        
        
        implemented_loss = ('vanilla')
        assert args.advloss in implemented_loss
        if args.advloss =='vanilla':
            self.loss_fn = self.vanilla_loss
        
        
    def build_model(self):
        
        implemented_networks = ('dcgan') #,'resnet', 'stylegan', 'stylegan2')
        
        assert self.model_name  in implemented_networks
       
        if self.model_name  == 'dcgan':
            self.D = dcgan.Discriminator(self.image_size)
            self.G = dcgan.Generator(self.image_size)
            
        self.D = self.D.to(self.device)
        self.G = self.G.to(self.device)

        
    def train(self, data_loader, args):
        losses = []
        checkpoints = []
        
        self.lr = 2e-4 #self.cfg['lr']
        epochs = 30 # self.cfg['epochs']
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        
        iteration = 0
        
        self.session_dir = ospj('./sessions',f'{self.model_name}_{self.advloss}')
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            os.makedirs(ospj(self.session_dir,'ckpts'))
            os.makedirs(ospj(self.session_dir,'fake_images'))
            os.makedirs(ospj(self.session_dir,'logs'))
            
            
        if len(glob(ospj(self.session_dir,'ckpts/D_*')))!=0:
            print('Loading checkpoints ...')
            lst = glob(ospj(self.session_dir,'ckpts/D_*'))
            iteration = sorted([int(re.findall("\d+",i)[-1]) for i in lst])[-1]
            self.load(iteration)
            
            
        for epoch in range(epochs):
            for imgs in data_loader:
                self.D.train()
                self.G.train()
                imgs = imgs.to(self.device)
                d_loss, g_loss = self.train_step(imgs, iteration)
                iteration +=1
                
                
                if (iteration+1) % args.ckpt_every == 0:
                    print(f"Saving Checkpoints ...")
                    self.save_ckpts(iteration+1)
                    
            
                if (iteration+1) % args.print_every == 0:
                    print(f"Iterations:{iteration+1} [D loss: {d_loss:.4f}] [G loss: {g_loss:.4f}]")

    #             if (iteration+1) % args.eval_every == 0:
    #                 print("%d/%d [D loss: %f] [G loss: %f]" % (epoch + 1, epochs, d_loss, g_loss))


                if (iteration+1) % args.sample_every == 0:
                    print(f"Iterations:{iteration+1} generated image is saved")
                    self.sample_images(self.G, iteration+1)
            
        return losses, checkpoints
    
    def train_step(self, imgs, iteration):
        z = torch.randn(imgs.size(0), self.image_size*8, 1,1, device=self.device)
        gen_imgs = self.G(z)
        
        if self.advloss =='vanilla':
            real = torch.ones(imgs.size(0), 1, device=self.device, dtype=torch.float)
            fake = torch.zeros(imgs.size(0), 1, device=self.device, dtype=torch.float)
            d_loss = self.loss_fn(self.D(imgs),real) + self.loss_fn(self.D(gen_imgs),fake)
            g_train_term = 1
            
        elif self.advloss =='wasserstein':
            #gradient_penalty = self.compute_gradient_penalty(self.D, imgs.data, gen_imgs.data)
            gradient_penalty = 0
            d_loss = -1 * torch.mean(self.D(imgs)) + torch.mean(self.D(gen_imgs)) +10 * gradient_penalty
            g_train_term = 5
        
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        gl = None
        
        
        if iteration%g_train_term==0:
#             for p in self.D.parameters():
#                 p.requires_grad = False

#             self.G.zero_grad()

            gen_imgs = self.G(z)
            
            
            if self.advloss =='vanilla':
                g_loss = self.loss_fn(self.D(gen_imgs),real)
            
            elif self.advloss =='wasserstein':
                g_loss = - torch.mean(self.D(gen_imgs))
            
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            gl = g_loss.item()
        return d_loss.item(), gl
    
    
    def vanilla_loss(self, logits, targets):
        loss = F.binary_cross_entropy(logits, targets)
        return loss
    
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = torch.ones(d_interpolates.size()).to(self.device).requires_grad_(False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def save_ckpts(self, iteration):
        torch.save({'iteration': iteration,'D_state_dict': self.D.state_dict()}, ospj(self.session_dir,f'ckpts/D_{iteration}.pth'))
        torch.save({'iteration': iteration,'G_state_dict': self.G.state_dict()}, ospj(self.session_dir,f'ckpts/G_{iteration}.pth'))
        torch.save({'iteration': iteration,'d_optimizer_state_dict': self.d_optimizer.state_dict()}, ospj(self.session_dir,f'ckpts/optim_d_{iteration}.pth'))
        torch.save({'iteration': iteration,'g_optimizer_state_dict': self.g_optimizer.state_dict()}, ospj(self.session_dir,f'ckpts/optim_g_{iteration}.pth'))

        
    def load(self, iteration):
        checkpoint = torch.load(ospj(self.session_dir,f'ckpts/D_{iteration}.pth'))
        self.D.load_state_dict(checkpoint['D_state_dict'])
        checkpoint = torch.load(ospj(self.session_dir,f'ckpts/G_{iteration}.pth'))
        self.G.load_state_dict(checkpoint['G_state_dict'])
        checkpoint = torch.load(ospj(self.session_dir,f'ckpts/optim_d_{iteration}.pth'))
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        checkpoint = torch.load(ospj(self.session_dir,f'ckpts/optim_g_{iteration}.pth'))
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        print(f'Load from iteration:{iteration}')
        
        
        
    def sample_images(self, generator, iteration):

        nrow=4
        
        z = torch.randn(nrow**2, self.image_size*8, 1, 1, device=self.device)

        gen_imgs = generator(z).cpu().detach()

        gen_imgs = 0.5 * (gen_imgs + 1.0)
        
        image_grid = vutils.make_grid(gen_imgs, padding=2, nrow=nrow,normalize=True)
        vutils.save_image(image_grid, ospj(self.session_dir,f'fake_images/iter_{iteration}.png'))


