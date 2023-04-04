from denoising_diffusion_pytorch_uncond_cpu import Unet, GaussianDiffusion, Trainer
import os

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


results_dir ='C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/results/uncond/low_res/'

# Config
image_size = 32
timesteps = 1000
train_batch_size = 128

config = {
    "train_batch_size": train_batch_size,
    "timesteps": timesteps,
    "image_size": image_size,
    "results_dir": results_dir
}
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = timesteps,           # number of steps
    sampling_timesteps = 1,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/total/train_B/',
    train_batch_size = train_batch_size,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = results_dir,
    config=config
)
mkdirs(results_dir)

config = {
    "train_batch_size": train_batch_size,
    "timesteps": timesteps,
    "image_size": image_size
}
trainer.train()