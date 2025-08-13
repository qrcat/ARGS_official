from models.basic_svae import SVAE, VAEGAR
from models.dataset import MergeGaussianDataModule

import lightning as L
from tqdm import tqdm, trange
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# we need to train a VAE first
svae = SVAE()
# svae = SVAE.load_from_checkpoint("log_svae/lightning_logs/version_0/checkpoints/epoch=1920-step=1921.ckpt")


dataset = MergeGaussianDataModule(
    "/data/workspace/ARGS/datasets",
    max_seq=8192, permute=True,
)

trainer = L.Trainer(
    default_root_dir="log_svae", 
    limit_train_batches=100, max_epochs=10000, log_every_n_steps=1)
trainer.fit(svae, datamodule=dataset)
# trainer.test(svae, datamodule=dataset)
