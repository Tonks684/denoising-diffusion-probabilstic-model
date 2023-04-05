import argparse
import os
import torch
from util import mkdirs

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArguementParser()
        self.initialized = False
    
    def initialize(self):
        self.parser.add_arguement()
        #Experiment Details
        self.parser.add_arguement('--name',type=str,default='experiment_name')
        #Generator
        self.parser.add_arguement('--init_dim', type=int, default=64, help='initial conv layer output channels')
        self.parser.add_arguement('--dim_mults', type=tuple, default= (1,2,4,8), help='output channel at each encoder layer is --dim*dim_mults[i]')
        self.parser.add_arguement('--channels',type=int,default=1,help='channel of input if using self_condition this will be automatically changed to 2')
        self.parser.add_arguement('--self_condition',type=bool,default=True, help='whether to condition on noise and additonal info')
        self.parser.add_arguement('--resenet_block_groups', type=int, default=8, help='total number of resent blocks')
        self.parser.add_arguement('--learned_variance', type=bool, default=False, help='whether to learn variance as well as mu')    
        self.parser.add_arguement('--learned_sinusoidal_cond', type=bool, default=False)
        self.parser.add_arguement('--learned_fourier_features', type=bool, default=False)
        self.parser.add_arguement('--learned_sinusoidal_dim', type=int, default=16)
        #Diffusion
        self.parser.add_arguement('--image_size',type=int,default=32, help='size you want to train on (if smaller than original image_size a crop will be taken)')
        self.parser.add_arguement('--sampling_timesteps',type=int,default=None)
        self.parser.add_arguement('--original_image_size',type=int,default=1080, help='size of input image')
        self.parser.add_arguement('--timesteps',type=int,default=1000, help='number of diffusion steps')
        self.parser.add_arguement('--loss_type',type=str,default='l1', help='loss type either l1 or l2')
        self.parser.add_arguement('--objective',type=str,default='pred_noise', help='pred_noise,pred_x0, pred_v')
        self.parser.add_arguement('--beta_schedule',type=str,default='sigmoid', help='sigmoid, linear or cosine')
        self.parser.add_arguement('--schedule_fn_kwargs',type=dict,default={})
        self.parser.add_arguement('--p2_loss_weight_gamma',type=int,default=0., help='p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended')
        self.parser.add_arguement('--p2_loss_weight_k',type=int,default=1)
        self.parser.add_arguement('--ddim_sampling_eta',type=int,default=0)
        self.parser.add_arguement('--auto_normalize',type=bool,default=False)
        self.parser.add_arguement('--x_self_cond',type=bool,default=None)

        #Dataset
        self.parser.add_arguement('--path_to_imgs', type=str,help='/path/to/imgs')
        self.parser.add_arguement('--augment_horizontal_flip', type=bool,default=False)
        self.parser.add_arguement('--convert_image_to', type=bool,default=None)
        self.parser.add_arguement('--shuffle', type=bool,default=False, help='whether to sample')
        self.parser.add_arguement('--num_workers', type=int,default=0, help='if running on local cpu 0 if HPC gpu go with 4')
        self.parser.add_arguement('--exts', type=list,default=['jpg', 'jpeg', 'png', 'tiff'], help='image extensions')
        
        #Training
        self.parser.add_arguement('--folder_A',type=str, help='/path/to/image/folder/A/')
        self.parser.add_arguement('--folder_B',type=str, help='/path/to/image/folder/B/')
        self.parser.add_arguement('--train_batch_size',type=int,default=16, help='batch size')
        self.parser.add_arguement('--gradient_accumulate_every',type=int,default=1)
        self.parser.add_arguement('--augment_horizontal_flip',type=bool,default=False)
        self.parser.add_arguement('--train_lr',default=8e-5, help='learning rate')
        self.parser.add_arguement('--train_num_steps',type=int,default=100000, help='number of steps')
        self.parser.add_arguement('--train_batch_size',type=int,default=16, help='batch size')
        self.parser.add_arguement('--ema_update_every',type=int,default=10, help='exponential moving average log update schedule')
        self.parser.add_arguement('--adam_betas',type=tuple,default=(0.9,0.99), help='Adam Beta values')
        self.parser.add_arguement('--ema_decay',type=int,default=16, help='exponential moving average decay')
        self.parser.add_arguement('--save_and_sample_every',type=int,default=200, help='number of steps after to save milestone and sample')
        self.parser.add_arguement('--num_samples',type=int,default=5, help='number of samples per sample')
        self.parser.add_arguement('--results_dir', type=str, help='path to save samples')
        self.parser.add_arguement('--results_dir', type=str, help='path to save samples')
        self.parser.add_arguement('--amp', type=bool,default=True, help='AMP Prcision')
        self.parser.add_arguement('--fp16', type=bool,default=True, help='Floating point 17')
        self.parser.add_arguement('--split_batches', type=bool,default=True, help='Split batches')
        self.parser.add_arguement('--convert_image_to', type=bool,default=None, help='convert image in get_transform')
        self.parser.add_arguement('--calculate_fid', type=bool,default=True, help='Calculate FID between target and samples')
        self.parser.add_arguement('--inception_block_idx', type=int,default=2048)

        # To add
        # --continue_train
        self.initialized = True

    def parse(self,save=True):
        if not self.initialized:
            self.initialized
        self.opt = self.parser.parse.parse_args()
        args = vars(self.opt)
        
        #Print all Options
        print('----------- Options -----------')
        for k, v in sorted(args.items()):
            print(f'{str(k)}: {str(v)}')
        print('----------- End -----------')
        #Save Options to disk
        
        expr_dir = os.path.join(self.opt.results_dir, self.opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('----------- Options -----------\n')
            for k, v in sorted(args.items()):
                opt.write(f'{str(k)}: {str(v)}')
            opt_file.write('----------- End -----------\n')
        return self.opt

    
        
        
        
        
        
        
        
