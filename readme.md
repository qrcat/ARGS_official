# ARGS: Auto-Regressive Gaussian Splatting via Parallel Progressive Next-Scale Prediction

[Quanyuan Ruan†](https://qrcat.github.io/)<sup>1</sup>, Kewei Shi<sup>2</sup>, [Jiabao Lei†](https://jblei.site/)<sup>3</sup>, Xifeng Gao*<sup>4</sup>, Xiaoguang Han*<sup>3</sup>

<sup>1</sup>South China University of Technology, <sup>2</sup>The University of Hong Kong, <sup>3</sup>The Chinese University of Hong Kong, Shenzhen, <sup>4</sup>Lightspeed

<sup>†</sup> Equal Contribution, <sup>*</sup> Corresponding authors

This is the official repository for ARGS(CVPRF 2026)



## Install
```bash
conda create -n args python=3.10 -y
conda activate args
# install torch >= 2.5.1 from https://pytorch.org/get-started/locally/
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# install requirements
pip install -r requirements.txt
```

## Example
```bash
python example.py
```

## Main Usage 

### Generate Data
```bash
python example_block.py
# or
python build_merge_list.py --input dataset --output dataset --workers 1
```

### Training

```bash
python train.py --dataset dataset/dataset --pattern "*block.pkl" --batch_size 1 --num_workers 1 --shuffle --logger wandb --model base_s_192 --devices 0 --accumulate_grad 1
```
### Eval
```bash
python train.py --dataset dataset/dataset --pattern "*block.pkl" --model base_s_192 --devices 0 --eval  --checkpoint log/args_args/kpj7jpb3/checkpoints/epoch=99-step=100.ckpt
```