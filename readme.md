
# 生成数据

## 生成modelsplat与shapesplat数据

```bash
python build_merge_list.py --shapesplat_ply xxx --modelsplat_ply xxx --output xxx
```

## 数据增强

对颜色\位置等信息进行增强(单个文件)

```bash
python enhance_data.py --input xxx.ply --output xxx.ply
```

## 渲染modelsplat与shapesplat数据(不参与训练,只为了可视化)

会在output地址下建立文件夹输出结果,需要安装```gsplat```

modelsplat

```bash
python render_modelsplat.py --modelsplat_ply xxx
```

shapesplat

```bash
python render_shapesplat.py --shapesplat_ply xxx
```

渲染单个GS文件

```bash
python render_single.py --path xxx.ply
```

# 训练模型

## 训练VQVAE

```bash
python train_vqvae.py
```

## 基于量化后的分类训练Transformer

```bash
python train_transformer1.py
```

# 一些工具

```pgs/__init__.py```简化算法的核心

```pgs/merge.py```将两个高斯merge的函数,建议使用```merge_gaussian_moments```或者```merge_gaussian_inv```,是论文中的一个核心创新点,在```pgs/__init__.py```有使用.


```models/transformer.py```transformer的代码

```models/svqvae.py```VQVAE的代码


```utils/args.py```解码

```utils/gaussian.py```求协方差与从协方差计算高斯的尺度\旋转的工具. numpy的计算没问题,但需注意torch的四元数是xyzw还是wxyz(还没有验证),高斯的定义是wxyz.

```utils/general.py```一些数学函数

```utils/io.py```输入输出

```utils/quantize.py```量化

```utils/quaternion.py```抄的pytorch3d的四元数

```utils/render.py```渲染的一些工具

```utils/shs.py```球谐转RGB与RGB转球谐
