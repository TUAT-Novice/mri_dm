import torch
import torch.nn as nn
import math
from abc import abstractmethod


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings through cosine and sine function.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    @abstractmethod
    def foward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, t_emb=None, c_emb=None, mask=None):
        for layer in self:
            if (isinstance(layer, TimestepBlock)):
                x = layer(x, t_emb, c_emb, mask)
            else:
                x = layer(x)
        return x


def norm_layer(channels):
    return nn.GroupNorm(16, channels)


# Residual block
class Residual_block(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, class_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.class_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_channels, out_channels)
        )

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t, c, mask):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        `c` has shape `[batch_size, class_dim]`
        `mask` has shape `[batch_size, ]`
        """
        h = self.conv1(x)
        emb_t = self.time_emb(t)
        emb_c = self.class_emb(c) * mask[:, None]
        h += (emb_t[:, :, None, None, None] + emb_c[:, :, None, None, None])
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W, D = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W * D).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W, D)
        h = self.proj(h)
        return h + x


# upsample
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2) # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(self.up(x))


# downsample
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels
        self.down = nn.MaxPool3d(kernel_size=2, stride=2) # nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(self.down(x))


class UnetModel(nn.Module):
    def __init__(self,
                 in_channels=3,
                 model_channels=64,
                 out_channels=3,
                 num_res_blocks=2,
                 attention_resolutions=(16,),
                 dropout=0,
                 channel_mult=(1, 2, 2, 2, 2),
                 conv_resample=True,
                 num_heads=4,
                 num_mod=10
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_mod = num_mod

        # time embedding
        time_emb_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # class embedding
        class_emb_dim = model_channels
        self.class_emb = nn.Embedding(num_mod, class_emb_dim)

        # input layers
        self.input_layers = TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1))

        # down blocks
        self.down_features = nn.ModuleList([])
        self.down_blocks = nn.ModuleList([])
        down_block_channels = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            layers = []
            for i in range(num_res_blocks):
                # res block
                layers.append(Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout))
                # attn block
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if i == num_res_blocks - 1:
                    down_block_channels.append(ch)
                    # down block
                    self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, ch * mult)))
                    ds *= 2
                    ch *= mult
            self.down_features.append(TimestepEmbedSequential(*layers))


        # middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            Residual_block(ch, ch, time_emb_dim, class_emb_dim, dropout)
        )

        # up blocks
        self.up_features = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(channel_mult[::-1]):
            layers = []
            for i in range(num_res_blocks):
                # up blcok
                if i == 0:
                    self.up_blocks.append(Upsample(ch, ch // mult))
                    ds //= 2
                    ch //= mult
                # res block
                layers.append(Residual_block(ch + down_block_channels.pop() if i == 0 else ch, ch, time_emb_dim, class_emb_dim, dropout))
                if ds in attention_resolutions:
                    # attn block
                    layers.append(AttentionBlock(ch, num_heads))
            self.up_features.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps, c, mask):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param c: a 1-D batch of classes.
        :param mask: a 1-D batch of conditioned/unconditioned.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        # time step and class embedding
        t_emb = self.time_emb(timestep_embedding(timesteps, dim=self.model_channels))
        c_emb = self.class_emb(c)

        # down stage
        h = self.input_layers(x)
        for down_fea, down in zip(self.down_features, self.down_blocks):
            h = down_fea(h, t_emb, c_emb, mask)
            hs.append(h)
            h = down(h)

        # middle stage
        h = self.middle_blocks(h, t_emb, c_emb, mask)

        # up stage
        for up_fea, up in zip(self.up_features, self.up_blocks):
            h = torch.cat([up(h), hs.pop()], dim=1)
            h = up_fea(h, t_emb, c_emb, mask)
        return self.out(h)
