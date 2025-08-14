C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
    
def data_enhance(self, data):
    # position enhance by scale
    std = torch.mean(data[..., 7:10], dim=-1)
    xyz_delta = torch.randn_like(data[..., :3]) * std[..., None] / 20
    # opacity enhance by itself
    opacity_delta = torch.randn_like(data[..., 3:4]) * data[..., 3:4] / 20 # PSNR >= 20.0
    # color jitter
    jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    
    rgb = SH2RGB(data[..., 4:7])
    rgb = rgb.permute(2, 0, 1)
    rgb = jitter(rgb)
    rgb = rgb.permute(1, 2, 0)
    shs = RGB2SH(rgb) # convert back to sh
    # scale jitter
    scale_delta = torch.randn_like(data[..., 7:10]) * data[..., 7:10] / 20
    # rotation jitter
    
    quaternion_delta = torch.randn_like(data[..., 10:14]) * data[..., 10:14] / 10
