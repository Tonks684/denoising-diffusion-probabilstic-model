from denoising_diffusion_pytorch_cond_cpu import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    self_condition=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 1,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/total/train_A/',
    'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/total/train_B/',
    train_batch_size = 1,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    results_folder = 
    'C:/Users/stonks/OneDrive - The Alan Turing Institute/Desktop/Cell_GSK_Dataset/',
)
# print(
trainer.train()