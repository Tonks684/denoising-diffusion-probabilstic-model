from denoising_diffusion_pytorch_uncond import Unet, GaussianDiffusion, Trainer
import os
from base_options import BaseOptions
import utils


# Parse baseoptions and any overwrites
opt = BaseOptions().parse()

# # Initialise Unet, Diffusion 
model = Unet(opt)
diffusion = GaussianDiffusion(model,opt)
num_params = utils.print_params(model)

# # W&D Logging
config = vars(opt)
config.update({'number_of_params': num_params})

# W&D Init
trainer = Trainer(diffusion,opt)

# Begin Training
trainer.train()