from models.data import SimpleData
from models.gtransformer import GTransformer
from utils.io import activated_gs2gs, save_ply
from utils.local import to_global
import torch


dataset = SimpleData()

model = GTransformer(14, 192, 12, 8, 0.1)
model.load_state_dict(torch.load('log/lightning_logs/version_3/checkpoints/epoch=26-step=7182.ckpt')['model_state_dict'])
model.eval()

before = 8
upstep = 20

now_gs, next_gs_split, new_gs, embedd = dataset[before]

now_gs_save = activated_gs2gs(now_gs)
save_ply(f'{before}.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])

cu_seqlens_kv = torch.tensor([0, embedd.shape[0]], dtype=torch.int32)
for i in range(upstep):
    cu_seqlens_gs = torch.tensor([0, now_gs.shape[0]], dtype=torch.int32)

    now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv = now_gs.cuda(), next_gs_split.cuda(), new_gs.cuda(), embedd.cuda(), cu_seqlens_gs.cuda(), cu_seqlens_kv.cuda()

    with torch.no_grad():
        split, dense = model(now_gs, now_gs[..., :3], embedd, cu_seqlens_gs, cu_seqlens_kv)

    split_mask = torch.squeeze(split > 0, dim=1)
    now_gs_split = now_gs[split_mask]
    
    new_gs_pred = dense[split_mask].view(-1, 2, 14)
    new_gs_pred = to_global(now_gs_split, new_gs_pred)
    
    if split_mask.all():
        now_gs = new_gs_pred.view(-1, 14)
    elif split_mask.any():
        # breakpoint()
        now_gs = now_gs[~split_mask]
        now_gs = torch.cat([now_gs, new_gs_pred.view(-1, 14)], dim=0)
    else:
        i -= 1
        break

now_gs_save = activated_gs2gs(now_gs)
save_ply(f'{before}to{before+i+1}.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])