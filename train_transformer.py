import lightning as L
from tqdm import tqdm, trange
import torch
# from models.svqvae import SVQVAE
from models.transformer import ARGSTransformer
# from pgs import PGS
# from pgs.utils import load_ply, save_ply
# from pgs.combine import GSItem, softplus_np, inv_softplus_np

# from models.gar import ByteGAR
# from models.svqvae import SVQVAE
from models.dataset import MergeGaussianDataModule, ARGaussianDataModule
# from models.svar import SGAR, NativeSGAR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

svar = ARGSTransformer(
    
)

dataset = ARGaussianDataModule(
    "/data/workspace/ARGS/datasets",
    vqvae="log_svqvae/lightning_logs/version_20/checkpoints/epoch=203-step=204.ckpt",
    permute=True, batch_size=2,
)

trainer = L.Trainer(limit_train_batches=100, max_epochs=10000, log_every_n_steps=1)
trainer.fit(svar, datamodule=dataset)
