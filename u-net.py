
import torch.nn as nn

from course.UNet_utils import ResidualConvBlock, DownBlock, UpBlock

class ResidualConvBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super().init()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.group_size = group_size
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
        )
    def forward(self, x):
        return self.conv_block(x) + x


class UNet(nn.Module):
    def __init__(self, T, img_ch, img_size, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=10):
        super().__init__()
        self.T = T
        up_chs = down_chs[::-1]
        latent_image_size = img_size // 4
        small_group_size = 8
        big_group_size = 32
        
        # Inital convolution
        self.down0 = ResidualConvBlock(img_ch, down_chs[0], small_group_size)

        # Downsample
        self.down1 = DownBlock(down_chs[0], down_chs[1], big_group_size)
        self.down2 = DownBlock(down_chs[1], down_chs[2], big_group_size)
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        # Embeddings
        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2] * latent_image_size**2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], t_embed_dim),
        )
        
        self.time_emb_proj = nn.Sequential(
            nn.Linear(t_embed_dim, down_chs[1]),
            nn.GELU(),
            nn.Linear(down_chs[1], down_chs[1]),
        )
        
        self.up0 = nn.ModuleList([
            ResidualConvBlock(down_chs[1] + up_chs[0], up_chs[0], small_group_size),
            ResidualConvBlock(up_chs[0] + down_chs[0], up_chs[0], small_group_size),
        ])  
        
        self.up1 = UpBlock(up_chs[0], up_chs[1], big_group_size)    
        self.up2 = UpBlock(up_chs[1], up_chs[2], big_group_size)
        
        self.out_conv = nn.Conv2d(up_chs[2], img_ch, kernel_size=1, padding=0)
        
    def forward(self, x, t, c, c_mask):
        t_emb = self.time_emb_proj(self.dense_emb(t))
        x = self.down0(x)
        x = self.down1(x, t_emb)
        
def main():