import argparse
import logging
import os
import torch
from datasets.main import get_data_loader 
from utils import load_yaml
from adagan import ADAGAN


def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg_path', type=str, default='./configs/ada_dcgan_cifar10.yaml', help='config_path')
    
    #dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--image_size', type=int, default=64, help='Use image_size to decide number of layers in the arch.')
    
    parser.add_argument('--arch', type=str, default='dcgan', help='architecture')
    
    
    parser.add_argument('--ckpt_every', type=int, default=1000, help='ckpt_every')
    parser.add_argument('--sample_every', type=int, default=5000, help='sample_every')
    parser.add_argument('--eval_every', type=int, default=1000, help='eval_every')
    parser.add_argument('--print_every', type=int, default=1000, help='print_every')
    
    #train
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='print_every')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--advloss', type=str, default='vanilla', help='adversarial loss')
    parser.add_argument('--reg', type=str, default=None, help='regressor')
    
    
    return parser.parse_args()
    
    
def main():
    args = parse_args()
    
    # Get configuration
#     cfg = load_yaml(args.cfg_path)
        
    # define network 
#     model = ADAGAN(cfg)
    model = ADAGAN(args)
    model.build_model()
    
    # load train & test set
    dataloader = get_data_loader(args)
   
    # train 
    model.train(dataloader, args)
    

if __name__ == '__main__':
    main()