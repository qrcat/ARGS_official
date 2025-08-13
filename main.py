import lightning as L
from tqdm import tqdm, trange
import torch

# from pgs import PGS
# from pgs.utils import load_ply, save_ply
# from pgs.combine import GSItem, softplus_np, inv_softplus_np

from models.vqvae import VQVAE
from models.dataset import ScaleGaussianDataModule


device = 'cuda' if torch.cuda.is_available() else 'cpu'

vqvae = VQVAE(z_channels=512, test_mode=False)
dataset = ScaleGaussianDataModule("/data/workspace/ARGS/datasets")

trainer = L.Trainer(limit_train_batches=100, max_epochs=10000, log_every_n_steps=1)
trainer.fit(vqvae, datamodule=dataset)
