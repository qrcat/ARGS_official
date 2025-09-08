from .basic import ScaleGaussianSplat
from .sin_encoder import GaussianSinEncoder, SinPositionEncoding
from utils.general import accuracy, top_p_sampling
import math

import lightning as L
import torch
import torch.nn as nn
from torch.nn import functional as F

# from dataset import get_shifted_sequence
from models.nanogpt import Block, LayerNorm, configure_optimizers
from tqdm import tqdm
import warnings

class ARGSTransformer(L.LightningModule):

    def __init__(
            self,
            gs_dims: int = 14,
            gs_nums: int = 2,
            # ==================== model params ====================
            n_layer: int = 24,
            n_head: int = 16,
            n_embd: int = 768,
            # for pretraining 0 is good, for finetuning try 0.1+
            dropout: float = 0.0, 
            # do we use bias inside LayerNorm and Linear layers?
            bias: bool = False, 
            # context_size of the transformer, for good performance
            # block_size > max sequence length
            block_size: int = 65536,
            # ======================================================
            weight_decay: float = 1e-1, 
            learning_rate: float = 1e-4, # max learning rate
            beta1: float = 0.9,
            beta2: float = 0.95,
            device_type: str = 'cuda',
        ):
        super().__init__()

        self.save_hyperparameters()
        
        sin_dims = 16
        self.scale_gs = ScaleGaussianSplat()
        self.sin_encoder = GaussianSinEncoder(sin_dims, True, True, True, True, embed_q=False)
        
        # embed the first token
        self.first_layer = nn.Linear(self.sin_encoder.out_dim, n_embd)
        # embed the other token
        self.input_layer = nn.Linear(self.sin_encoder.out_dim*gs_nums, n_embd)
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(bias, block_size, dropout, n_embd, n_head) for _ in range(n_layer)]),
            ln_f=LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, gs_dims*gs_nums, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        self.pos_act = lambda x: torch.tanh(x)
        self.opa_act = lambda x: torch.sigmoid(x)
        self.scl_act = lambda x: 0.1 * F.softplus(x)
        self.rot_act = lambda x, dim: F.normalize(x, dim=dim)

    def training_step(self, batch, batch_idx):
        source, target, indice, bhmask = batch
        
        pred = self(target[:, :1], source, indice)

        loss_x = torch.nn.functional.l1_loss(self.pos_act(pred[..., :3]), source[..., :3]) * 2
        loss_o = torch.nn.functional.l1_loss(self.opa_act(pred[..., 3:4]), source[..., 3:4]) * 1.5
        loss_f = torch.nn.functional.l1_loss(pred[..., 4:7], source[..., 4:7].clip(min=-1.772453850905516, max=1.772453850905516)) / 1.772453850905516
        loss_s = torch.nn.functional.l1_loss(self.scl_act(pred[..., 7:10]), source[..., 7:10]) * 10
        
        dot_product = torch.sum(self.rot_act(pred[..., 10:], dim=-1) * source[..., 10:], dim=-1).abs()
        loss_q = 1 - dot_product.mean()

        loss = loss_x + loss_o + loss_f + loss_s + loss_q

        self.log_dict({
            'loss/train_loss': loss,
            'loss/loss_x': loss_x,
            'loss/loss_o': loss_o,
            'loss/loss_f': loss_f,
            'loss/loss_s': loss_s,
            'loss/loss_q': loss_q,
        })

        return loss

    def compute_SNR(self, x, y, peak=None):
        """Compute Signal-to-Noise Ratio (SNR) between x and y."""
        
        signal = y**2 if peak is None else torch.full_like(y, peak**2)
        noises = (x - y)**2
        return 20 * (torch.log10((signal) - torch.log10(noises.clip(min=1e-5)))).mean().item()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        source, target, indice, mask = batch
        
        pred = self(target[:, :1], source, indice)

        true_x, true_o, true_f, true_s, true_q = source.split([3, 1, 3, 3, 4], dim=-1)
        # prepare the predict
        pred_x, pred_o, pred_f, pred_s, pred_q = pred.split([3, 1, 3, 3, 4], dim=-1)
        
        pred_x = self.pos_act(pred_x)
        pred_o = self.opa_act(pred_o)
        pred_f = pred_f
        pred_s = self.scl_act(pred_s)
        pred_q = self.rot_act(pred_q, dim=-1)

        pred_x = pred_x[mask]
        pred_o = pred_o[mask]
        pred_f = pred_f[mask]
        pred_s = pred_s[mask]
        pred_q = pred_q[mask]

        true_x = true_x[mask]
        true_o = true_o[mask]
        true_f = true_f[mask]
        true_s = true_s[mask]
        true_q = true_q[mask]

        metrics = {
            "eval/l1_xyz": torch.nn.functional.l1_loss(pred_x, true_x).item(),
            "eval/SNR_xyz": self.compute_SNR(pred_x, true_x, peak=1.0),
            "eval/sphere_distance_xyz": (torch.norm(pred_x-true_x, dim=-1)).mean().item(),

            "eval/l1_opacity": torch.nn.functional.l1_loss(pred_o, true_o).item(),
            "eval/SNR_opacity": self.compute_SNR(pred_o, true_o, peak=1.0),

            "eval/l1_feature": torch.nn.functional.l1_loss(pred_f, true_f).item(),
            "eval/SNR_feature": self.compute_SNR(pred_f, true_f, peak=1.78),

            "eval/l1_scale": torch.nn.functional.l1_loss(pred_s, true_s).item(),
            "eval/SNR_scale": self.compute_SNR(pred_s, true_s),

            "eval/quat_distance": 1-(torch.sum(pred_q * true_q, dim=-1).abs()).mean().item(),
        }

        self.log_dict(metrics, sync_dist=True)


    def forward(self, init, gaussian, indice, kv_cache=None, mask_cache=None):
        use_kv_cache = kv_cache is not None
        device = gaussian.device
        b, t = indice.shape
        assert t <= self.hparams.block_size, f"Cannot forward sequence of length {t}, block size is only {self.hparams.block_size}"

        init = self.scale_gs.forward(init)
        init = self.sin_encoder(init)
        tok_fst = self.first_layer(init)

        gaussian = self.scale_gs.forward(gaussian[:, :-1])    
        gaussian = self.sin_encoder(gaussian)
        gaussian = gaussian.flatten(-2, -1)
        tok_app = self.input_layer(gaussian)

        tok_emb = torch.cat([tok_fst, tok_app], dim=1)
        
        # position embedding
        if kv_cache is not None and kv_cache[0].numel():
            pos = kv_cache[0].shape[-2]  # kv_cache of shape: num_layers * (2, B, nh, T, hs)
            pos_emb = self.transformer.wpe.weight[None, pos]  # 1 x n_embd
            mask = mask_cache.index_select(2, torch.LongTensor([pos]).to(pos_emb.device))[:, :, :, :pos + 1]
        else:
            pos = torch.tensor([i for i in range(t)], dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            mask = None

        sum_emb = tok_emb + pos_emb[None]
        # print('shapes:', tok_emb.shape, pos_emb.shape, coord_emb.shape)
        x = self.transformer.drop(sum_emb)

        # apply multiple transformer blocks
        new_kv_cache = []
        kv_cache = kv_cache or [None] * self.hparams.n_layer

        for block, kv_cache_layer in zip(self.transformer.h, kv_cache):
            x, new_kv = block(x, kv_cache_layer, mask)
            new_kv_cache.append(new_kv)

        x = self.transformer.ln_f(x)

        output = self.lm_head(x)

        return output.view(b, t, self.hparams.gs_nums, self.hparams.gs_dims)


    @torch.no_grad()
    def generate(self, idx, fin, fout, tokenizer, max_new_tokens=10000, temperature=1.0, top_k=None, top_p=0.9, use_kv_cache=False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if use_kv_cache and (max_new_tokens + idx.shape[-1] - 1) > self.hparams.block_size:
            # print(f"Cannot generate more than {self.config.block_size} tokens with kv cache, setting max new tokens to {self.config.block_size - idx.shape[-1]}")
            max_new_tokens = self.hparams.block_size - idx.shape[-1]

        kv_cache = (
            [torch.empty(2, 0, device=idx.device, dtype=idx.dtype) for _ in range(self.hparams.n_layer)]
            if use_kv_cache
            else None
        )
        mask_cache = None
        if use_kv_cache:
            ones = torch.ones((self.hparams.block_size, self.hparams.block_size), device=idx.device, dtype=torch.bool)
            mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

        current_fin = fin
        current_fout = fout
        one_t = torch.LongTensor([1]).to(fin.device)
        for iteration in range(max_new_tokens):

            if not use_kv_cache or (iteration == 0 and idx.shape[-1] > 1):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.hparams.block_size else idx[:, -self.hparams.block_size:]
                fin_cond = current_fin if current_fin.size(1) <= self.hparams.block_size else current_fin[:, -self.hparams.block_size:]
                fout_cond = current_fout if current_fout.size(1) <= self.hparams.block_size else current_fout[:, -self.hparams.block_size:]
                fout_cond = torch.from_numpy(get_shifted_sequence(fout_cond[0].cpu().numpy())).to(idx_cond.device).unsqueeze(0)
            else:
                idx_cond = idx[:, -1:]
                fin_cond = current_fin[:, -1:]
                fout_cond = current_fout[:, -1:]  # note: don't need shifting since we assume block_size is huge enough to not need shifting
            # forward the model to get the logits for the index in the sequence
            logits, kv_cache = self(idx_cond, fin_cond, fout_cond, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # TODO: Introduce hard constraints

            # sample from the distribution
            # apply softmax to convert logits to (normalized) probabilities
            if top_p is not None:
                idx_next = top_p_sampling(logits, top_p)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            last_fin_cond = current_fin[0, -1]
            if last_fin_cond == self.finemb_size - 1 or (iteration == 0 and idx.shape[-1] == 2):
                current_fin = torch.cat((current_fin, (3 * one_t[0]).unsqueeze(0).unsqueeze(0)), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0)), dim=1)
            else:
                current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0)), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1]).unsqueeze(0).unsqueeze(0)), dim=1)
            if idx_next == 1:
                return idx
        return None


    @torch.no_grad()
    def generate_with_beamsearch(self, idx, fin, fout, tokenizer, max_new_tokens=10000, use_kv_cache=False, beam_width=6):

        backup_beams = []
        backup_beam_prob = []
        max_new_tokens = self.hparams.block_size - idx.shape[-1]

        kv_cache = (
            [torch.empty(2, 0, device=idx.device, dtype=idx.dtype) for _ in range(self.hparams.n_layer)]
            if use_kv_cache
            else None
        )

        mask_cache = None

        if use_kv_cache:
            ones = torch.ones((self.hparams.block_size, self.hparams.block_size), device=idx.device, dtype=torch.bool)
            mask_cache = torch.tril(ones).unsqueeze(0).unsqueeze(0)

        current_fin = fin
        current_fout = fout
        one_t = torch.LongTensor([1]).to(fin.device)

        idx = idx.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        current_fin = current_fin.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        current_fout = current_fout.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)

        logits, kv_cache = self(idx, fin, fout, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)

        vocabulary_size = logits.shape[-1]
        probabilities, top_k_indices = logits[0, 0, :].squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

        next_chars = top_k_indices.reshape(-1, 1)
        idx = torch.cat((idx, next_chars), axis=-1)

        last_fin_cond = current_fin[0, -1]  # same for all beams
        if last_fin_cond == self.finemb_size - 1 or (idx.shape[-1] == 2):
            current_fin = torch.cat((current_fin, (3 * one_t[0]).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
        else:
            current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            current_fout = torch.cat((current_fout, current_fout[0, -1].unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
        
        for iteration in tqdm(range(max_new_tokens - 1), desc='beam_search'):
            if not use_kv_cache:
                idx_cond = idx
                fin_cond = current_fin
                fout_cond = current_fout
            else:
                idx_cond = idx[:, -1:]
                fin_cond = current_fin[:, -1:]
                fout_cond = current_fout[:, -1:]

            # forward the model to get the logits for the index in the sequence
            logits, kv_cache = self(idx_cond, fin_cond, fout_cond, tokenizer, kv_cache=kv_cache if use_kv_cache else None, mask_cache=mask_cache)

            next_probabilities = logits.log_softmax(-1)

            next_probabilities = next_probabilities.reshape((-1, beam_width, next_probabilities.shape[-1]))
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim=1)
            probabilities, top_k_indices = probabilities.topk(k=beam_width, axis=-1)
            next_indices = torch.remainder(top_k_indices, vocabulary_size).flatten().unsqueeze(-1)
            best_candidates = (top_k_indices / vocabulary_size).long()
            best_candidates += torch.arange(idx.shape[0] // beam_width, device=idx.device).unsqueeze(-1) * beam_width
            idx = idx[best_candidates].flatten(end_dim=-2)
            for block_idx in range(len(kv_cache)):
                kv_cache[block_idx] = kv_cache[block_idx][:, best_candidates.flatten(), :, :, :]
            idx = torch.cat((idx, next_indices), axis=1)

            # update fin and fout
            last_fin_cond = current_fin[0, -1]  # same for all beams
            if last_fin_cond == self.finemb_size - 1 or (idx.shape[-1] == 2):
                current_fin = torch.cat((current_fin, (3 * one_t[0]).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
                current_fout = torch.cat((current_fout, (current_fout[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            else:
                current_fin = torch.cat((current_fin, (current_fin[0, -1] + 1).unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
                current_fout = torch.cat((current_fout, current_fout[0, -1].unsqueeze(0).unsqueeze(0).expand((beam_width, -1))), dim=1)
            
            amax = probabilities.flatten().argmax()
            if idx[amax, -1] == 1:
                return idx[amax: amax + 1, :]
            for beam_idx in range(beam_width):
                if idx[beam_idx, -1] == 1:
                    backup_beams.append(idx[beam_idx: beam_idx + 1, :])
                    backup_beam_prob.append(probabilities[0, beam_idx].item())
        return None

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for n,p in self.named_parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self):
        betas = (self.hparams.beta1, self.hparams.beta2)
        return configure_optimizers(
            self.named_parameters(), 
            self.hparams.weight_decay, 
            self.hparams.learning_rate, 
            betas, 
            self.hparams.device_type
        )
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is None:
            warnings.warn("No lr_schedulers found")
            optimizer = self.optimizers().optimizer
            for i, param in enumerate(optimizer.param_groups):
                self.log(f"lr/last_lr_{i}", param['lr'], sync_dist=True)
        elif isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                self.log_scheduler_lr(lr_scheduler)
        else:
            self.log_scheduler_lr(lr_schedulers)
    
    def log_scheduler_lr(self, lr_scheduler):
        for i in range(len(lr_scheduler.get_last_lr())):
            self.log(f"lr/last_lr_{i}", lr_scheduler.get_last_lr()[i], sync_dist=True)
