
import math
import copy
from pathlib import Path
from random import random
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tifffile import imsave
from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA 

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from version import __version__
import wandb
# wandb.init()
import matplotlib.pyplot as plt
from utils import *

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        # for the batch and channels split image into 4 and return
        # number of images, flattened 4 sections, new_height, new_width
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization

    Replacement to standard Conv layer
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    """
    Used to apply groupnorm before the attention layer
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds
# parameters of NN are shared across time (noise level)
# spe encode t (inspired by Transformers) makes NN "know" at which t it is operating, for every image in batch


class SinusoidalPosEmb(nn.Module):
    """
    Takes tensor of shape (batch_size,1) as input (ie noise level of several noisy images  in batch
    and turns into tensor of shape (btach_size,dim) with dim being dimensionality
    of positional embedding. This is then added to each residual block.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    """
    dim_head: square root of dimensions(k)
    heads: number of scaled-dot product attention heads
    Difference compared to regular attention is the time and memory
    constratins scaler linarly in the sequence length, as oppossed
    to quadratically for regulaer attention
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # query, key and value
        # By performing a convolution2d on x and then splitting into chunks on batch
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # Split image into parts based on self.heads
        # Produce same index but x and y are concatenated
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    """
    Added inbetween conv blocks. Again building block from transformer
    This attention is regular multi-head self-attention
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale
        # 'b h d i' dimension of q (d is summattion index) (repeated indices inputs will be multiplied and products wil be outputs)
        # 'b h d j' dimension of k (d is summattion index as it is not in the output)
        # 'b h i j' dimension of sim (free indices)

        sim = einsum('b h d i, b h d j -> b h i j', q, k) #
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        opt):

        super().__init__()

        # determine dimensions

        self.channels = opt.channels
        self.self_condition = opt.self_condition
        input_channels = opt.channels * (2 if opt.self_condition else 1)
        init_dim = default(None, opt.init_dim)
        
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: opt.init_dim * m, opt.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'OUtput channels are {in_out}')

        block_klass = partial(ResnetBlock, groups = opt.resnet_block_groups)

        # time embeddings

        time_dim = opt.init_dim * 4

        self.random_or_learned_sinusoidal_cond = opt.learned_sinusoidal_cond or opt.random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(opt.learned_sinusoidal_dim, opt.random_fourier_features)
            fourier_dim = opt.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(opt.init_dim)
            fourier_dim = opt.init_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = opt.channels * (1 if not opt.learned_variance else 2)
        self.out_dim = default_out_dim

        self.final_res_block = block_klass(opt.init_dim * 2, opt.init_dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(opt.init_dim, self.out_dim, 1)

    def forward(self, x, time, cond_A = None):
        if self.self_condition:
            cond_A = default(cond_A, lambda: torch.zeros_like(x))
            x = torch.cat([x, cond_A], dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    # b here is total number of betas
    b, *_ = t.shape
    # torch.gather gathers values along an axis specified by dim (-1) at index t
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        opt,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.model = model

        self.channels = self.model.channels
        # Binary variable for self conditioning
        self.self_condition = self.model.self_condition

        self.image_size = opt.image_size

        self.objective = opt.objective
        self.original_image_size = opt.original_image_size
        assert opt.objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 ' \
            '(predict image start) or pred_v (predict v [v-parameterization ' \
            'as defined in appendix D of progressive distillation paper, ' \
            'used in imagen-video successfully])'

        if opt.beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif opt.beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif opt.beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {opt.beta_schedule}')

        # DDPM formulas
        # q(xt|xt-1)= N(xt; sqrt(alpha_t)*xt-1,Beta_t*I)
        # q(x1:T|x0)= cum_dot_product_1toT(q(xt|xt-1))

        betas = beta_schedule_fn(opt.timesteps, **opt.schedule_fn_kwargs)
        # Beta from 1 to T (change in variance schedule based on no. timesteps)
        alphas = 1. - betas
        # .cumprod returns array at_1, at_1*at_2, at_1*at_2*at_3,...at_n! of all alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Padding last dimension by 1 on each side
        # This alphas_cumprod up until T-1 then at T it is padded with 0s
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(opt.timesteps)
        self.loss_type = opt.loss_type

        # sampling related parameters

        self.sampling_timesteps = default(opt.sampling_timesteps, opt.timesteps) # default num sampling timesteps to number of timesteps at training
        # 1000 timesteps
        # self.sampling_timesteps are inference timesteps
        assert self.sampling_timesteps <= opt.timesteps
        # If sampling timesteps are smaller than timesteps (1000) then is ddim sampling else not
        self.is_ddim_sampling = self.sampling_timesteps < opt.timesteps
        self.ddim_sampling_eta = opt.ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # q(xt|xt-1)= N(xt; sqrt(alpha_t)*xt-1,Beta_t*I)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod)) # alpha_t above
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod)) # 1 - alpha_t above. I thought it was 1 - beta which is alpha not 1 - alpha
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))# log(1 - alpha)_t above
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod)) # 1 /alpha_t above
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))# 1 /alpha_t -1 above

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (opt.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -opt.p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False
        
        self.normalize = normalize_to_neg_one_to_one if opt.auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if opt.auto_normalize else identity
        self.tensor2img = unnormalize_tensor_to_img 

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, cond_A = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, cond_A)
        # maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond_A = None, clip_denoised = False):
        preds = self.model_predictions(x, t, cond_A=cond_A)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_A = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond_A = cond_A, clip_denoised = False)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False, cond_A=None):
        """
        Full sampling back from T to 1
        cond_A = Conditional input that is concatenated with x_start (Gaussian Noise at T and then reduced noise as t goes to)
        """
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        
        x_start = img # Complete Noise

        for t in tqdm(reversed(range(0, self.num_timesteps)), 
        desc = 'sampling loop time step', total = self.num_timesteps):
        # for t in reversed(range(1,1000)):
            # x_start at 0 is Total Total Noise
            # what we need is total noise conditioned on BF
            # self_cond = x_start if self.self_condition else None
            # cat_A_xstart = torch.cat(cond_A,x_start,dim=1)
            # print(t)
            img_tminus1, x_start = self.p_sample(img, t, cond_A)
            imgs.append(img_tminus1)
        print(f'sampling loops {len(imgs)}')
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        
        # ret_0to1 = self.unnormalize(ret)
        ret_16bit = self.tensor2img(ret)

        return ret_16bit

    @torch.no_grad()
    def ddim_sample(self, shape,return_all_timesteps = False,cond_A=None):

        """
        Skipping some sampling steps faster computation (intuition)
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist())) # [T-1, T-2,... -1]
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # Complete noise
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long) #create tensor of size (bactch,) and fill in with time so if T is 1000 T-1 999 torch.long enforces int64
            self_cond = x_start if self.self_condition else None # if self_condition x_start = None for first iter
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_A=cond_A, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        ret = self.tensor2img(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False, cond_A=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # sample_fn = self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size),cond_A=cond_A, return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, cond_A=None, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # x_self_cond = None
        # if self.self_condition and random() < 0.5:
        #     with torch.no_grad():
        #         x_self_cond = self.model_predictions(x, t).pred_x_start
        #         x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, cond_A)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean') #

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()


    def forward(self, img, cond_A=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        # return self.p_losses(img, t, *args, **kwargs)
        return self.p_losses(
            img, t, cond_A=cond_A, noise = None
            )

# dataset classes

class Dataset(Dataset):
    def __init__(
        self,
        opt,
        channel=None
    ):
        super().__init__()
        if channel == "A":
            self.folder = opt.folder_A
            self.paths = [p for ext in opt.exts for p in
                        Path(f'{opt.folder_A}').glob(f'*.{ext}')]
        else:
            self.folder = opt.folder_B
            self.paths = [p for ext in opt.exts for p in
                        Path(f'{opt.folder_B}').glob(f'*.{ext}')]
        self.image_size = opt.image_size
        self.original_image_size = opt.original_image_size
        
        self.transform = T.Compose([
            # Crop image
            # T.Lambda(lambda img: __crop(img, params['crop_pos'],self.image_size)),
            # T.Lambda(maybe_convert_fn), # Identidy only as convert_image_to = None
            # T.Resize(image_size),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(size=(self.image_size,self.image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('F')
        img = img.point(lambda p: p*(1/65535))
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        opt
    ):
        super().__init__()

        #wandb config and init
        wandb.init()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = opt.split_batches,
            mixed_precision = 'fp16' if opt.fp16 else 'no',
        )

        self.accelerator.native_amp = opt.amp

        # model

        self.model = diffusion_model

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if opt.calculate_fid:
            assert opt.inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opt.inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = opt.num_samples
        self.save_and_sample_every = opt.save_and_sample_every

        self.batch_size = opt.train_batch_size
        self.gradient_accumulate_every = opt.gradient_accumulate_every

        self.train_num_steps = opt.train_num_steps
        self.image_size = diffusion_model.image_size
        self.original_image_size = diffusion_model.original_image_size
        # dataset and dataloader

        self.ds_A = Dataset(opt,channel="A")
        dl_A = DataLoader(self.ds_A,
                        batch_size = opt.train_batch_size,
                        shuffle = opt.shuffle, 
                        pin_memory = True,
                        num_workers = opt.num_workers)

        dl_A = self.accelerator.prepare(dl_A)
        self.dl_A = cycle(dl_A)

        self.ds_B = Dataset(opt,channel="B")
        dl_B = DataLoader(self.ds_B,
                        batch_size = opt.train_batch_size,
                        shuffle = opt.shuffle,
                        pin_memory = True,
                        num_workers = opt.num_workers)
        

        dl_B = self.accelerator.prepare(dl_B)
        self.dl_B = cycle(dl_B)
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = opt.train_lr, betas = opt.adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = opt.ema_decay, update_every = opt.ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(opt.results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            return self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):
        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
                initial = self.step,
                total = self.train_num_steps,
                disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):

                    data_A = next(self.dl_A).to(device)
                    data_B = next(self.dl_B).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data_B, cond_A=data_A)
                        wandb.log({'loss':loss.item()})
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=1,[2]))
                            sample = self.ema.ema_model.sample(batch_size=self.batch_size,cond_A=data_A)
                            # sample = self.model.sample(batch_size=self.batch_size,cond_A=data_A)
                       
                        ## Save locally
                        # for item in all_images_list:
                        #     for i in item:
                        #         print(f'list item {np.asarray(i).shape} item dtype {i.dtype}')
                        # all_images = torch.cat(all_images_list, dim = 0)
                        # for index,img in enumerate(sample):
                            # imsave(self.results_folder, f'/stacks/sample_stack_index_{index}.tiff', img.astype(np.float32), imagej=True)
                        
                        
                        # Save W&B
                        wandb.log({'Input': wandb.Image(data_A[0])})
                        wandb.log({'Target1': wandb.Image(data_B[0])})
                        wandb.log({'Sample_target1': wandb.Image(plt.imshow(sample[0],cmap='gray'))})
                        wandb.log({'Sample Histogram': wandb.Histogram(sample[0])})
                        
                        # whether to calculate fid
                        if exists(self.inception_v3):
                            fid_score = self.fid_score(real_samples = data_B, fake_samples = all_images)
                            accelerator.print(f'fid_score: {fid_score}')
                            wandb.log({'fid':fid_score})
                        


                pbar.update(1)

        accelerator.print('training complete')
