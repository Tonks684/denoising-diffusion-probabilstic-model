import argparse
import os
import torch
from utils import *
import wandb
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        #Experiment Details
        self.parser.add_argument('--name',type=str,default='bf_dna_low_res_test3')
        self.parser.add_argument('--results_folder', type=str, 
            default='C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/results/cond/low_res/',help='path to save samples')
        self.parser.add_argument('--folder_A',type=str,
            default='C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/64x64/train_A/', help='/path/to/image/folder/A/')
        self.parser.add_argument('--folder_B',type=str,
            default='C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/64x64/train_B/', help='/path/to/image/folder/B/')        
        #Generator
        self.parser.add_argument('--init_dim', type=int, default=64, help='initial conv layer output channels')
        self.parser.add_argument('--dim_mults', type=tuple, default= (1,1,2,2), help='output channel at each encoder layer is --dim*dim_mults[i]')
        self.parser.add_argument('--channels',type=int,default=1,help='channel of input if using self_condition this will be automatically changed to 2')
        self.parser.add_argument('--self_condition',type=bool,default=True, help='whether to condition on noise and additonal info')
        self.parser.add_argument('--resnet_block_groups', type=int, default=4, help='total number of resent blocks')
        self.parser.add_argument('--learned_variance', type=bool, default=False, help='whether to learn variance as well as mu')    
        self.parser.add_argument('--learned_sinusoidal_cond', type=bool, default=False)
        self.parser.add_argument('--learned_fourier_features', type=bool, default=False)
        self.parser.add_argument('--random_fourier_features', type=bool, default=False)
        self.parser.add_argument('--learned_sinusoidal_dim', type=int, default=16)
        #Diffusion
        self.parser.add_argument('--image_size',type=int,default=64, help='size you want to train on (if smaller than original image_size a crop will be taken)')
        self.parser.add_argument('--sampling_timesteps',type=int,default=1000)
        self.parser.add_argument('--original_image_size',type=int,default=64, help='size of input image')
        self.parser.add_argument('--timesteps',type=int,default=1000, help='number of diffusion steps')
        self.parser.add_argument('--loss_type',type=str,default='l1', help='loss type either l1 or l2')
        self.parser.add_argument('--objective',type=str,default='pred_noise', help='pred_noise,pred_x0, pred_v')
        self.parser.add_argument('--beta_schedule',type=str,default='sigmoid', help='sigmoid, linear or cosine')
        self.parser.add_argument('--schedule_fn_kwargs',type=dict,default={})
        self.parser.add_argument('--p2_loss_weight_gamma',type=int,default=0., help='p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended')
        self.parser.add_argument('--p2_loss_weight_k',type=int,default=1)
        self.parser.add_argument('--ddim_sampling_eta',type=int,default=0)
        self.parser.add_argument('--auto_normalize',type=bool,default=False)
        self.parser.add_argument('--x_self_cond',type=bool,default=None)
        #Dataset
        self.parser.add_argument('--path_to_imgs', type=str,help='/path/to/imgs')
        self.parser.add_argument('--augment_horizontal_flip', type=bool,default=False)
        self.parser.add_argument('--shuffle', type=bool,default=False, help='whether to sample')
        self.parser.add_argument('--num_workers', type=int,default=0, help='if running on local cpu 0 if HPC gpu go with 4')
        self.parser.add_argument('--exts', type=list,default=['jpg', 'jpeg', 'png', 'tiff'], help='image extensions')
        #Training
        self.parser.add_argument('--train_batch_size',type=int,default=4, help='batch size')
        self.parser.add_argument('--gradient_accumulate_every',type=int,default=1)
        self.parser.add_argument('--train_lr',default=8e-5, help='learning rate')
        self.parser.add_argument('--train_num_steps',type=int,default=100000, help='number of steps')
        self.parser.add_argument('--ema_update_every',type=int,default=10, help='exponential moving average log update schedule')
        self.parser.add_argument('--adam_betas',type=tuple,default=(0.9,0.99), help='Adam Beta values')
        self.parser.add_argument('--ema_decay',type=int,default=16, help='exponential moving average decay')
        self.parser.add_argument('--save_and_sample_every',type=int,default=5000, help='number of steps after to save milestone and sample')
        self.parser.add_argument('--num_samples',type=int,default=1, help='number of samples per sample')
        self.parser.add_argument('--amp', type=bool,default=False, help='AMP Prcision')
        self.parser.add_argument('--fp16', type=bool,default=False, help='Floating point 16')
        self.parser.add_argument('--split_batches', type=bool,default=False, help='Split batches')
        self.parser.add_argument('--convert_image_to', type=bool,default=None, help='convert image in get_transform')
        self.parser.add_argument('--calculate_fid', type=bool,default=False, help='Calculate FID between target and samples')
        self.parser.add_argument('--inception_block_idx', type=int,default=2048)

        # To add
        # --continue_train
        
        self.initialized = True

    def parse(self,save=True):
        if not self.initialized:
            self.initialized
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        
        #Print all Options
        print('----------- Options -----------')
        for k, v in sorted(args.items()):
            print(f'{str(k)}: {str(v)}')
        print('----------- End ---------------')
        #Save Options to disk
        
        expr_dir = os.path.join(self.opt.results_folder, self.opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('----------- Options -----------\n')
            for k, v in sorted(args.items()):
                opt_file.write(f'{str(k)}: {str(v)}')
            opt_file.write('----------- End -----------\n')
        return self.opt

    
        
        
        
        
        
        
        
