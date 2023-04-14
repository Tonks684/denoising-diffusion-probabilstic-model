from denoising_diffusion_pytorch_cond_cpu import Unet, GaussianDiffusion, Trainer
import os
from base_options import BaseOptions
import wandb
import utils
# Parse baseoptions and any overwrites
opt = BaseOptions().parse()
# Initialise Unet, Diffusion and Trainer 
model = Unet(opt)
utils.print_params(model)
diffusion = GaussianDiffusion(model,opt)
trainer = Trainer(diffusion,opt)

# Begin Training
wandb.init()
wandb.config.update(vars(opt))
wandb.config.update({'number_of_params': utils.print_params(model)})
wandb.watch(model)
wandb.watch(diffusion)
trainer.train()