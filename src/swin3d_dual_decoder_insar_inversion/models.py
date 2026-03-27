from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition_3d(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    b, t, h, w, c = x.shape
    wt, wh, ww = window_size
    x = x.view(b, t // wt, wt, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wt * wh * ww, c)


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    b: int,
    t: int,
    h: int,
    w: int,
    c: int,
) -> torch.Tensor:
    wt, wh, ww = window_size
    x = windows.view(b, t // wt, h // wh, w // ww, wt, wh, ww, c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, t, h, w, c)


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_windows, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(b_windows, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_windows // n_w, n_w, self.num_heads, n_tokens, n_tokens)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n_tokens, n_tokens)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n_tokens, dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int] = (3, 4, 4),
        shift_size: tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def _attn_mask(self, t: int, h: int, w: int, device: torch.device) -> torch.Tensor | None:
        if all(s == 0 for s in self.shift_size):
            return None
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size
        img_mask = torch.zeros((1, t, h, w, 1), device=device)
        count = 0
        t_slices = (slice(0, -wt), slice(-wt, -st), slice(-st, None))
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -ww), slice(-ww, -sw), slice(-sw, None))
        for ts in t_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, ts, hs, ws, :] = count
                    count += 1
        mask_windows = window_partition_3d(img_mask, self.window_size).view(-1, wt * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        wt, wh, ww = self.window_size
        pad_t = (wt - t % wt) % wt
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
        _, tp, hp, wp, c = x.shape
        shortcut = x
        shifted = x
        if any(s > 0 for s in self.shift_size):
            shifted = torch.roll(shifted, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        windows = window_partition_3d(shifted, self.window_size)
        windows = self.norm1(windows)
        attn_mask = self._attn_mask(tp, hp, wp, x.device)
        attn_windows = self.attn(windows, mask=attn_mask)
        shifted = window_reverse_3d(attn_windows, self.window_size, b, tp, hp, wp, c)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted
        x = shortcut + x
        x_flat = x.view(-1, c)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.view(b, tp, hp, wp, c)
        x = x[:, :t, :h, :w]
        return x.permute(0, 4, 1, 2, 3).contiguous()


class SwinStage3D(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: tuple[int, int, int]):
        super().__init__()
        shift = tuple(s // 2 for s in window_size)
        self.blocks = nn.ModuleList(
            [
                SwinBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if idx % 2 == 0 else shift,
                )
                for idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 32, patch_size: tuple[int, int, int] = (2, 4, 4)):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchMerging3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.reduction = nn.Conv3d(in_dim, out_dim, kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x)


class PatchExpand3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: tuple[int, int, int] = (1, 2, 2)):
        super().__init__()
        self.expand = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(x)


class DualDecoderFrequencySeparatedSwinUNet3D(nn.Module):
    def __init__(
        self,
        base_dim: int = 32,
        time_patch: int = 2,
        spatial_patch: int = 4,
        num_heads: int = 4,
        window_size: tuple[int, int, int] = (3, 4, 4),
        merge_scale: tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        patch_size = (time_patch, spatial_patch, spatial_patch)
        self.low_embed = PatchEmbed3D(in_channels=1, embed_dim=base_dim, patch_size=patch_size)
        self.high_embed = PatchEmbed3D(in_channels=1, embed_dim=base_dim, patch_size=patch_size)
        self.enc = SwinStage3D(dim=base_dim * 2, depth=2, num_heads=num_heads, window_size=window_size)
        self.merge = PatchMerging3D(in_dim=base_dim * 2, out_dim=base_dim * 4, scale=merge_scale)
        self.bottleneck = SwinStage3D(dim=base_dim * 4, depth=2, num_heads=num_heads, window_size=window_size)
        self.s0_expand = PatchExpand3D(in_dim=base_dim * 4, out_dim=base_dim * 2, scale=merge_scale)
        self.sg_expand = PatchExpand3D(in_dim=base_dim * 4, out_dim=base_dim * 2, scale=merge_scale)
        self.s0_fuse = nn.Conv3d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.sg_fuse = nn.Conv3d(base_dim * 4, base_dim * 2, kernel_size=1)
        self.s0_decoder = SwinStage3D(dim=base_dim * 2, depth=2, num_heads=num_heads, window_size=window_size)
        self.sg_decoder = SwinStage3D(dim=base_dim * 2, depth=2, num_heads=num_heads, window_size=window_size)
        self.s0_up = nn.ConvTranspose3d(base_dim * 2, base_dim, kernel_size=patch_size, stride=patch_size)
        self.sg_up = nn.ConvTranspose3d(base_dim * 2, base_dim, kernel_size=patch_size, stride=patch_size)
        self.s0_head = nn.Conv3d(base_dim, 1, kernel_size=1)
        self.sg_head = nn.Conv3d(base_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deformation = x[:, 0:1]
        low = F.avg_pool3d(deformation, kernel_size=3, stride=1, padding=1)
        high = deformation - low
        low_tokens = self.low_embed(low)
        high_tokens = self.high_embed(high)
        enc = self.enc(torch.cat([low_tokens, high_tokens], dim=1))
        bottleneck = self.bottleneck(self.merge(enc))

        s0_dec = self.s0_expand(bottleneck)
        s0_dec = self.s0_decoder(self.s0_fuse(torch.cat([s0_dec, enc], dim=1)))
        s0 = self.s0_head(self.s0_up(s0_dec))[:, :, -1]

        sg_dec = self.sg_expand(bottleneck)
        sg_dec = self.sg_decoder(self.sg_fuse(torch.cat([sg_dec, enc], dim=1)))
        sg = self.sg_head(self.sg_up(sg_dec))[:, :, -1]
        return torch.cat([s0, sg], dim=1)

