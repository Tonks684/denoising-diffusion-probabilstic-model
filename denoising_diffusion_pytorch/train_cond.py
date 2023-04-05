from denoising_diffusion_pytorch_cond_cpu import Unet, GaussianDiffusion, Trainer
import os
from base_options import BaseOptions

# Parse baseoptions and any overwrites
opt = BaseOptions().parse()
opt.results_dir ='C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/results/cond/low_res/'
opt.folder_A = 'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/total/train_A/',
opt.folder_B = 'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/total/train_B/',

# Initialise Unet, Diffusion and Trainer 
model = Unet(opt)
diffusion = GaussianDiffusion(model,opt)
trainer = Trainer(diffusion,opt)
# Begin Training
wandb.init(config=vars(opt))
trainer.train()