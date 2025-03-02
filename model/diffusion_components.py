import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
from inspect import isfunction
import math
from tqdm import tqdm


def exists(x):
    """Return true for x is not None."""
    return x is not None


def default(val, d):
    """Helper function"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    """Skip connection"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    """Upsample layer, a transposed convolution layer with stride=2"""
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    """Downsample layer, a convolution layer with stride=2"""
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    """Return sinusoidal embedding for integer time step."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Stack of convolution, normalization, and non-linear activation"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Stack of [conv + norm + act (+ scale&shift)], with positional embedding inserted <https://arxiv.org/abs/1512.03385>"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # Adding positional embedding to intermediate layer (by broadcasting along spatial dimension)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """Stack of [conv7x7 (+ condition(pos)) + norm + conv3x3 + act + norm + conv3x3 + res1x1]，with positional embedding inserted"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    """Apply normalization before 'fn'"""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class ConditionalEmbedding(nn.Module):
    """Return embedding for label and projection for text embedding"""

    def __init__(self, num_labels, embedding_dim, condition_type="instrument_family"):
        super(ConditionalEmbedding, self).__init__()
        if condition_type == "instrument_family":
            self.embedding = nn.Embedding(num_labels, embedding_dim)
        elif condition_type == "natural_language_prompt":
            self.embedding = nn.Linear(embedding_dim, embedding_dim, bias=True)
        else:
            raise NotImplementedError()

    def forward(self, labels):
        return self.embedding(labels)


class LinearCrossAttention(nn.Module):
    """Combination of efficient attention and cross attention."""

    def __init__(self, dim, heads=4, label_emb_dim=128, dim_head=32):
        super().__init__()
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

        # embedding for key and value
        self.label_key = nn.Linear(label_emb_dim, hidden_dim)
        self.label_value = nn.Linear(label_emb_dim, hidden_dim)

    def forward(self, x, label_embedding=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        if label_embedding is not None:
            label_k = self.label_key(label_embedding).view(b, self.heads, self.dim_head, 1)
            label_v = self.label_value(label_embedding).view(b, self.heads, self.dim_head, 1)

            k = torch.cat([k, label_k], dim=-1)
            v = torch.cat([v, label_v], dim=-1)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


def pad_to_match(encoder_tensor, decoder_tensor):
    """
    Pads the decoder_tensor to match the spatial dimensions of encoder_tensor.

    :param encoder_tensor: The feature map from the encoder.
    :param decoder_tensor: The feature map from the decoder that needs to be upsampled.
    :return: Padded decoder_tensor with the same spatial dimensions as encoder_tensor.
    """

    enc_shape = encoder_tensor.shape[2:]  # spatial dimensions are at index 2 and 3
    dec_shape = decoder_tensor.shape[2:]

    # assume enc_shape >= dec_shape
    delta_w = enc_shape[1] - dec_shape[1]
    delta_h = enc_shape[0] - dec_shape[0]

    # padding
    padding_left = delta_w // 2
    padding_right = delta_w - padding_left
    padding_top = delta_h // 2
    padding_bottom = delta_h - padding_top
    decoder_tensor_padded = F.pad(decoder_tensor, (padding_left, padding_right, padding_top, padding_bottom))

    return decoder_tensor_padded


def pad_and_concat(encoder_tensor, decoder_tensor):
    """
    Pads the decoder_tensor and concatenates it with the encoder_tensor along the channel dimension.

    :param encoder_tensor: The feature map from the encoder.
    :param decoder_tensor: The feature map from the decoder that needs to be concatenated with encoder_tensor.
    :return: Concatenated tensor.
    """

    # pad decoder_tensor
    decoder_tensor_padded = pad_to_match(encoder_tensor, decoder_tensor)
    # concat encoder_tensor and decoder_tensor_padded
    concatenated_tensor = torch.cat((encoder_tensor, decoder_tensor_padded), dim=1)
    return concatenated_tensor


class LinearCrossAttentionAdd(nn.Module):
    def __init__(self, dim, heads=4, label_emb_dim=128, dim_head=32):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.label_emb_dim = label_emb_dim
        self.dim_head = dim_head

        self.hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(self.dim, self.hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(self.hidden_dim, dim, 1), nn.GroupNorm(1, dim))

        # embedding for key and value
        self.label_key = nn.Linear(label_emb_dim, self.hidden_dim)
        self.label_query = nn.Linear(label_emb_dim, self.hidden_dim)


    def forward(self, x, condition=None):
        b, c, h, w = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # if condition exists，concat its key and value with origin
        if condition is not None:
            label_k = self.label_key(condition).view(b, self.heads, self.dim_head, 1)
            label_q = self.label_query(condition).view(b, self.heads, self.dim_head, 1)
            k = k + label_k
            q = q + label_q

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)



def linear_beta_schedule(timesteps):
    """
    Generate linearly spaced beta values.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def get_beta_schedule(timesteps):
    """
    Compute diffusion constants.

    Returns:
      sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_recip_alphas
    """
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance, sqrt_recip_alphas


def extract(a, t, x_shape):
    """
    Extract values from 1D tensor 'a' at indices 't' and reshape for broadcasting.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    Generate x_t by adding noise to x_start at timestep t.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise














